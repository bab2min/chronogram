#include <numeric>
#include <iostream>
#include <iterator>
#include "ChronoGramModel.h"
#include "IOUtils.h"
#include "ThreadPool.h"
#include "polynomials.hpp"

using namespace std;
using namespace Eigen;

#define DEBUG_PRINT(out, x) (out << #x << ": " << x << endl)


void ChronoGramModel::buildModel()
{
	const size_t V = vocabs.size();
	// allocate & initialize model
	in = MatrixXf::Zero(M, L * V);
	for (size_t v = 0; v < V; ++v)
	{
		in.block(0, v * L, M, 1) = MatrixXf::Random(M, 1) * (.5f / M);
	}

	out = MatrixXf::Random(M, V) * (.5f / M);

	timePrior = VectorXf::Zero(L);
	timePrior[0] = 1;
	normalizeWordDist(false);
	buildTable();
}

void ChronoGramModel::buildTable()
{
	// build Unigram Table for Negative Sampling
	vector<double> weights;
	transform(frequencies.begin(), frequencies.end(), back_inserter(weights), [](auto w) { return pow(w, 0.75); });
	unigramTable = discrete_distribution<uint32_t>(weights.begin(), weights.end());
	auto p = unigramTable.probabilities();
	unigramDist = vector<float>{ p.begin(), p.end() };
	wordScale = vector<float>(vocabs.size(), 1.f);
}

VectorXf ChronoGramModel::makeCoef(size_t L, float z)
{
	VectorXf coef = VectorXf::Zero(L);
	for (size_t i = 0; i < L; ++i)
	{
		coef[i] = poly::chebyshevTGet(i, 2 * z - 1);
	}
	return coef;
}

VectorXf ChronoGramModel::makeDCoef(size_t L, float z)
{
	VectorXf coef = VectorXf::Zero(L - 1);
	for (size_t i = 1; i < L; ++i)
	{
		coef[i - 1] = 2 * poly::chebyshevTDerived(i, 2 * z - 1);
	}
	return coef;
}

VectorXf ChronoGramModel::makeTimedVector(size_t wv, const VectorXf& coef) const
{
	return in.block(0, wv * L, M, L) * coef;
}

template<bool _Fixed>
float ChronoGramModel::inplaceUpdate(size_t x, size_t y, float lr, bool negative, const VectorXf& lWeight)
{
	auto outcol = out.col(y);
	VectorXf inSum = _Fixed ? in.col(x * L) : makeTimedVector(x, lWeight);
	float f = inSum.dot(outcol);
	float pr = logsigmoid(f * (negative ? -1 : 1)) * (1 - zeta);

	float d = ((negative ? 0 : 1) - sigmoid(f)) * (1 - zeta);
	float g = lr * d;

	VectorXf in_grad = g * outcol;
	VectorXf out_grad = g * inSum;
	outcol += out_grad;

	if (_Fixed || fixedWords.count(x))
	{
		in.col(x * L) += in_grad;
	}
	else
	{
		in.block(0, x*L, M, L) += in_grad * VectorXf{ lWeight.array() * vEta.array() }.transpose();
	}
	return pr;
}

template<bool _Fixed>
float ChronoGramModel::getUpdateGradient(size_t x, size_t y, float lr, bool negative, const VectorXf& lWeight,
	Eigen::DenseBase<Eigen::MatrixXf>::ColXpr xGrad, Eigen::DenseBase<Eigen::MatrixXf>::ColXpr yGrad)
{
	auto outcol = out.col(y);
	VectorXf inSum = _Fixed ? in.col(x * L) : makeTimedVector(x, lWeight);
	float f = inSum.dot(outcol);
	float pr = logsigmoid(f * (negative ? -1 : 1)) * (1 - zeta);
	if (!isfinite(pr))
	{
		lock_guard<mutex> lock(mtx);
		DEBUG_PRINT(cout, pr);
		DEBUG_PRINT(cout, f);
		throw exception();
	}

	float d = ((negative ? 0 : 1) - sigmoid(f)) * (1 - zeta);
	if (!isfinite(d))
	{
		lock_guard<mutex> lock(mtx);
		DEBUG_PRINT(cout, f);
		throw exception();
	}
	float g = lr * d;

	xGrad += g * outcol;
	yGrad += g * inSum;
	return max(pr, -100.f);
}

float ChronoGramModel::inplaceTimeUpdate(size_t x, float lr, const VectorXf& lWeight, 
	const vector<const VectorXf*>& randSampleWeight)
{
	/*
	s = int(1 - exp(-|a Tv|^2/2))
	P = (1 - exp(-|a.Tv|^2/2)) / s

	log P = log(1 - exp(-|a.Tv|^2/2)) - log(s)
	d log(1 - exp(-|a.Tv|^2/2)) / da = 1 / (1 - exp(-|a.Tv|^2/2)) (-exp(-|a.Tv|^2/2)) (-a.Tv) . Tv^T
		= (a.Tv . Tv^T / (exp(|a.Tv|^2/2) - 1))

	d log(s) / da = d log(s) / ds * ds / da = 1 / s * ds / da
	ds/da = int(a.Tv . Tv^T * exp(-|a.Tv|^2/2))
	d log P / da = (a.Tv . Tv^T / (exp(|a.Tv|^2/2) - 1)) - int(a.Tv . Tv^T * exp(-|a.Tv|^2/2)) / s
	*/
	auto atv = makeTimedVector(x, lWeight);
	float pa = max(getTimePrior(lWeight), 1e-5f);
	float expTerm = min(exp(-atv.squaredNorm() / 2 * lambda * pa), 1.f);
	float ll = log(1 - expTerm + 1e-5f);
	auto d = atv * lWeight.transpose() * expTerm / (1 - expTerm + 1e-5f) * lambda * pa;
	float s = 0;
	MatrixXf nd = MatrixXf::Zero(M, L);
	for (auto& r : randSampleWeight)
	{
		auto atv = makeTimedVector(x, *r);
		float pa = max(getTimePrior(*r), 1e-5f);
		float expTerm = min(exp(-atv.squaredNorm() / 2 * lambda * pa), 1.f);
		nd += atv * r->transpose() * expTerm * lambda * pa;
		s += 1 - expTerm;
	}
	s /= randSampleWeight.size();
	s = max(s, 1e-5f);
	nd /= randSampleWeight.size();
	ll -= log(s);
	in.block(0, x*L, M, L) -= -(d - nd / s) * lr * zeta;
	return ll * zeta;
}

float ChronoGramModel::getTimeUpdateGradient(size_t x, float lr, const VectorXf & lWeight,
	const vector<const VectorXf*>& randSampleWeight, Block<MatrixXf> grad)
{
	/*
	s = int(1 - exp(-|a Tv|^2/2))
    P = (1 - exp(-|a.Tv|^2/2)) / s

    log P = log(1 - exp(-|a.Tv|^2/2)) - log(s)
    d log(1 - exp(-|a.Tv|^2/2)) / da = 1 / (1 - exp(-|a.Tv|^2/2)) (-exp(-|a.Tv|^2/2)) (-a.Tv) . Tv^T
        = (a.Tv . Tv^T / (exp(|a.Tv|^2/2) - 1))

    d log(s) / da = d log(s) / ds * ds / da = 1 / s * ds / da
    ds/da = int(a.Tv . Tv^T * exp(-|a.Tv|^2/2))
    d log P / da = (a.Tv . Tv^T / (exp(|a.Tv|^2/2) - 1)) - int(a.Tv . Tv^T * exp(-|a.Tv|^2/2)) / s
	*/
	auto atv = makeTimedVector(x, lWeight);
	float pa = max(getTimePrior(lWeight), 1e-5f);
	float expTerm = min(exp(-atv.squaredNorm() / 2 * lambda * pa), 1.f);
	float ll = log(1 - expTerm + 1e-5f);
	if (!isfinite(ll))
	{
		lock_guard<mutex> lock(mtx);
		DEBUG_PRINT(cout, ll);
		DEBUG_PRINT(cout, expTerm);
		DEBUG_PRINT(cout, pa);
		throw exception();
	}
	auto d = atv * lWeight.transpose() * expTerm / (1 - expTerm + 1e-5f) * lambda * pa;
	float s = 0;
	MatrixXf nd = MatrixXf::Zero(M, L);
	for (auto& r : randSampleWeight)
	{
		auto atv = makeTimedVector(x, *r);
		float pa = max(getTimePrior(*r), 1e-5f);
		float expTerm = min(exp(-atv.squaredNorm() / 2 * lambda * pa), 1.f);
		nd += atv * r->transpose() * expTerm * lambda * pa;
		s += 1 - expTerm;
	}
	s /= randSampleWeight.size();
	s = max(s, 1e-5f);
	nd /= randSampleWeight.size();
	ll -= log(s);
	if (!isfinite(ll))
	{
		lock_guard<mutex> lock(mtx);
		DEBUG_PRINT(cout, ll);
		DEBUG_PRINT(cout, expTerm);
		DEBUG_PRINT(cout, pa);
		DEBUG_PRINT(cout, s);
		throw exception();
	}
	grad -= -(d - nd/s) * lr * zeta;
	return ll * zeta;
}


float ChronoGramModel::updateTimePrior(float lr, const VectorXf & lWeight, const vector<const VectorXf*>& randSampleWeight)
{
	/*
	s = int(1 - exp(-|a Tv|^2/2))
	P = (1 - exp(-|a.Tv|^2/2)) / s

	log P = log(1 - exp(-|a.Tv|^2/2)) - log(s)
	d log(1 - exp(-|a.Tv|^2/2)) / da = 1 / (1 - exp(-|a.Tv|^2/2)) (-exp(-|a.Tv|^2/2)) (-a.Tv) * Tv
		= (Tv * a.Tv / (exp(|a.Tv|^2/2) - 1))

	d log(s) / da = d log(s) / ds * ds / da = 1 / s * ds / da
	ds/da = int(Tv * a.Tv * exp(-|a.Tv|^2/2))
	d log P / da = (Tv * a.Tv / (exp(|a.Tv|^2/2) - 1)) - int(Tv * a.Tv * exp(-|a.Tv|^2/2)) / s
	*/
	float p = timePrior.dot(lWeight);
	float expTerm = min(exp(-p * p / 2), 1.f);
	float ll = log(1 - expTerm + 1e-5f);
	auto d = lWeight * p * expTerm / (1 - expTerm + 1e-5f);
	float s = 0;
	VectorXf nd = VectorXf::Zero(L);
	for (auto& r : randSampleWeight)
	{
		float dot = timePrior.dot(*r);
		float expTerm = min(exp(-dot * dot / 2), 1.f);
		nd += *r * dot * expTerm;
		s += 1 - expTerm;
	}
	s /= randSampleWeight.size();
	s = max(s, 1e-5f);
	nd /= randSampleWeight.size();
	ll -= log(s);
	timePrior -= -(d - nd/s) * lr;
	if (!isfinite(ll))
	{
		lock_guard<mutex> lock(mtx);
		DEBUG_PRINT(cout, ll);
		DEBUG_PRINT(cout, s);
		DEBUG_PRINT(cout, expTerm);
		DEBUG_PRINT(cout, p);
		throw exception();
	}
	return ll;
}

template<bool _Fixed, bool _GNgramMode>
void ChronoGramModel::trainVectors(const uint32_t * ws, size_t N, float timePoint,
	size_t windowLen, float start_lr, size_t nEpoch, size_t report)
{
	VectorXf coef = makeCoef(L, normalizedTimePoint(timePoint));
	float lr1 = max(start_lr * (1 - procWords / (totalWords + 1.f)), start_lr * 1e-4f);

	for (size_t i = 0; i < N; ++i)
	{
		const auto& x = ws[i];
		if (_GNgramMode && i != 0 && i != N - 1) continue;
		if (_GNgramMode && x == (uint32_t)-1) continue;

		size_t jBegin = 0, jEnd = N - 1;
		if (i > windowLen) jBegin = i - windowLen;
		if (i + windowLen < N) jEnd = i + windowLen;

		float llSum = 0, llUg = 0;
		size_t llCnt = 0;
		// update in, out vector
		if (zeta < 1)
		{
			for (auto j = jBegin; j <= jEnd; ++j)
			{
				if (i == j) continue;
				if (_GNgramMode && ws[j] == (uint32_t)-1) continue;
				float ll = inplaceUpdate<_Fixed>(x, ws[j], lr1, false, coef);
				assert(isfinite(ll));
				for (size_t k = 0; k < negativeSampleSize; ++k)
				{
					uint32_t ns = unigramTable(globalData.rg);
					while (ns == ws[j]) ns = unigramTable(globalData.rg);
					ll += inplaceUpdate<_Fixed>(x, ns, lr1, true, coef);
					assert(isfinite(ll));
				}
				llCnt++;
				llSum += ll;
			}
		}
		else
		{
			llCnt += jEnd - jBegin - 1;
		}

		if (!_Fixed && zeta > 0 && !fixedWords.count(x))
		{
			vector<VectorXf> randSampleWeight;
			vector<const VectorXf*> randSampleWeightP;
			for (size_t i = 0; i < temporalSample; ++i)
			{
				randSampleWeight.emplace_back(makeCoef(L, generate_canonical<float, 16>(globalData.rg)));
			}
			for (size_t i = 0; i < temporalSample; ++i)
			{
				randSampleWeightP.emplace_back(&randSampleWeight[i]);
			}
			llUg += inplaceTimeUpdate(x, lr1, coef, randSampleWeightP);
		}

		if (llCnt)
		{
			totalLLCnt += llCnt;
			totalLL += (llSum - llCnt * totalLL) / totalLLCnt;
			ugLL += (llUg - llCnt * ugLL) / totalLLCnt;
		}
	}

	procWords += _GNgramMode ? 0 : N;
	if (!_Fixed && report && (procWords - _GNgramMode ? 0 : N) / report < procWords / report)
	{
		float time_per_kword = (procWords - lastProcWords) / timer.getElapsed() / 1000.f;
		fprintf(stderr, "%.2f%% %.4f %.4f %.4f %.4f %.2f kwords/sec\n",
			procWords / (totalWords / 100.f), totalLL + ugLL, totalLL, ugLL,
			lr1, time_per_kword);
		lastProcWords = procWords;
		timer.reset();
	}
}

template<bool _Fixed, bool _GNgramMode>
void ChronoGramModel::trainVectorsMulti(const uint32_t * ws, size_t N, float timePoint,
	size_t windowLen, float start_lr, size_t nEpoch, size_t report, ThreadLocalData& ld)
{
	VectorXf coef = makeCoef(L, normalizedTimePoint(timePoint));

	float lr1 = max(start_lr * (1 - procWords / (totalWords + 1.f)), start_lr * 1e-4f);

	float llSum = 0, llUg = 0;
	size_t llCnt = 0;

	ld.updateOutIdx.clear();
	ld.updateOutMat = MatrixXf::Zero(M, negativeSampleSize * (windowLen + 1) * 2);
	MatrixXf updateIn = MatrixXf::Zero(M, 1);
	MatrixXf updateInBlock = MatrixXf::Zero(M, L);
	for (size_t i = 0; i < N; ++i)
	{
		const auto& x = ws[i];
		if (_GNgramMode && i != 0 && i != N - 1) continue;
		if (_GNgramMode && x == (uint32_t)-1) continue;

		size_t jBegin = 0, jEnd = N - 1;
		if (i > windowLen) jBegin = i - windowLen;
		if (i + windowLen < N) jEnd = i + windowLen;
		updateIn.setZero();

		// update in, out vector
		if (zeta < 1)
		{
			for (auto j = jBegin; j <= jEnd; ++j)
			{
				if (i == j) continue;
				if (_GNgramMode && ws[j] == (uint32_t)-1) continue;
				if (ld.updateOutIdx.find(ws[j]) == ld.updateOutIdx.end())
					ld.updateOutIdx.emplace(ws[j], ld.updateOutIdx.size());

				float ll = getUpdateGradient<_Fixed>(x, ws[j], lr1, false, coef,
					updateIn.col(0),
					ld.updateOutMat.col(ld.updateOutIdx[ws[j]]));
				assert(isfinite(ll));
				for (size_t k = 0; k < negativeSampleSize; ++k)
				{
					uint32_t ns = unigramTable(ld.rg);
					while (ns == ws[j]) ns = unigramTable(ld.rg);
					if (ld.updateOutIdx.find(ns) == ld.updateOutIdx.end())
						ld.updateOutIdx.emplace(ns, ld.updateOutIdx.size());

					ll += getUpdateGradient<_Fixed>(x, ns, lr1, true, coef,
						updateIn.col(0),
						ld.updateOutMat.col(ld.updateOutIdx[ns]));
					assert(isfinite(ll));
				}
				llCnt++;
				llSum += ll;
			}
		}
		else
		{
			llCnt += jEnd - jBegin - 1;
		}
		
		updateInBlock.setZero();
		if (!_Fixed && zeta > 0 && !fixedWords.count(x))
		{
			vector<VectorXf> randSampleWeight;
			vector<const VectorXf*> randSampleWeightP;
			for (size_t i = 0; i < temporalSample; ++i)
			{
				randSampleWeight.emplace_back(makeCoef(L, generate_canonical<float, 16>(ld.rg)));
			}
			for (size_t i = 0; i < temporalSample; ++i)
			{
				randSampleWeightP.emplace_back(&randSampleWeight[i]);
			}
			llUg += getTimeUpdateGradient(x, lr1, coef, 
				randSampleWeightP,
				updateInBlock.block(0, 0, M, L));
		}

		{
			lock_guard<mutex> lock(mtx);
			// deferred update
			if (_Fixed || fixedWords.count(x))
			{
				in.col(x * L) += updateIn.col(0);
			}
			else
			{
				in.block(0, x * L, M, L) += updateIn * VectorXf{ coef.array() * vEta.array() }.transpose();
			}

			if (!_Fixed && zeta > 0 && !fixedWords.count(x))
			{
				in.block(0, x * L, M, L) += updateInBlock;
			}

			for (auto& p : ld.updateOutIdx)
			{
				out.col(p.first) += ld.updateOutMat.col(p.second);
			}

		}
		ld.updateOutMat.setZero();
		ld.updateOutIdx.clear();
	}

	lock_guard<mutex> lock(mtx);
	if (llCnt)
	{
		totalLLCnt += llCnt;
		totalLL += (llSum - llCnt * totalLL) / totalLLCnt;
		ugLL += (llUg - llCnt * ugLL) / totalLLCnt;
	}
	procWords += _GNgramMode ? 0 : N;

	if (!_Fixed && report && (procWords - (_GNgramMode ? 0 : N)) / report < procWords / report)
	{
		float time_per_kword = (procWords - lastProcWords) / timer.getElapsed() / 1000.f;
		fprintf(stderr, "%.2f%% %.4f %.4f %.4f %.4f %.2f kwords/sec\n",
			procWords / (totalWords / 100.f), totalLL + ugLL, totalLL, ugLL,
			lr1, time_per_kword);
		lastProcWords = procWords;
		timer.reset();
	}
}

void ChronoGramModel::trainTimePrior(const float * ts, size_t N, float lr, size_t report)
{
	unordered_map<float, VectorXf> coefMap;
	vector<float> randSample(temporalSample);
	vector<const VectorXf*> randSampleWeight(temporalSample);
	for (size_t i = 0; i < N; ++i)
	{
		for (size_t r = 0; r < temporalSample; ++r)
		{
			randSample[r] = generate_canonical<float, 16>(globalData.rg);
			if (!coefMap.count(randSample[r])) coefMap.emplace(randSample[r], makeCoef(L, randSample[r]));
		}
		float c_lr = max(lr * (1 - procTimePoints / (totalTimePoints + 1.f)), lr * 1e-4f) * 0.1f;
		auto it = coefMap.find(ts[i]);
		if (it == coefMap.end()) it = coefMap.emplace(ts[i], makeCoef(L, normalizedTimePoint(ts[i]))).first;
		for (size_t r = 0; r < temporalSample; ++r)
		{
			randSampleWeight[r] = &coefMap.find(randSample[r])->second;
		}
		
		float ll = updateTimePrior(c_lr, it->second, randSampleWeight);
		procTimePoints++;
		timeLLCnt++;
		timeLL += (ll - timeLL) / timeLLCnt;
		if (report && procTimePoints % report == 0)
		{
			fprintf(stderr, "timePrior LL: %.4f\n", timeLL);
			/*for (size_t r = 0; r < L; ++r)
			{
				fprintf(stderr, "%.4f ", timePrior[r]);
			}
			fprintf(stderr, "\n");*/
		}
	}
}

void ChronoGramModel::normalizeWordDist(bool updateVocab)
{
	constexpr size_t step = 128;

	vector<VectorXf> coefs;
	for (size_t i = 0; i <= step; ++i)
	{
		coefs.emplace_back(makeCoef(L, i * (1.f - timePadding * 2) / step + timePadding));
	}

	float p = 0;
	for (size_t i = 0; i <= step; ++i)
	{
		p += 1 - exp(-pow(timePrior.dot(coefs[i]), 2) / 2);
	}
	p /= step + 1;
	timePriorScale = max(p, 1e-5f);
	if (updateVocab)
	{
		for (size_t v = 0; v < vocabs.size(); ++v)
		{
			float p = 0;
			for (size_t i = 0; i <= step; ++i)
			{
				p += 1 - exp(-makeTimedVector(v, coefs[i]).squaredNorm() / 2 * lambda * getTimePrior(coefs[i]));
			}
			p /= step + 1;
			wordScale[v] = p;
		}
	}
}

float ChronoGramModel::getTimePrior(const Eigen::VectorXf & coef) const
{
	return (1 - exp(-pow(timePrior.dot(coef), 2) / 2)) / timePriorScale;
}

float ChronoGramModel::getWordProbByTime(uint32_t w, const Eigen::VectorXf & timedVector, float tPrior) const
{
	return (1 - exp(-timedVector.squaredNorm() / 2 * lambda * tPrior)) / wordScale[w];
}

float ChronoGramModel::getWordProbByTime(uint32_t w, float timePoint) const
{
	auto coef = makeCoef(L, normalizedTimePoint(timePoint));
	return getWordProbByTime(w, makeTimedVector(w, coef), getTimePrior(coef));
}


void ChronoGramModel::buildVocab(const std::function<ReadResult(size_t)>& reader, size_t minCnt, size_t numWorkers)
{
	if (!numWorkers) numWorkers = thread::hardware_concurrency();

	float minT = INFINITY, maxT = -INFINITY;
	unordered_map<string, uint32_t> counter;
	if (numWorkers > 1)
	{
		vector<unordered_map<string, uint32_t>> counters(numWorkers - 1);
		{
			ThreadPool workers(numWorkers - 1, (numWorkers - 1) * 16);
			vector<string> words;
			for (size_t id = 0; ; ++id)
			{
				auto res = reader(id);
				if (res.stop) break;
				if (res.words.empty()) continue;
				minT = min(res.timePoint, minT);
				maxT = max(res.timePoint, maxT);
				words.insert(words.end(), make_move_iterator(res.words.begin()), make_move_iterator(res.words.end()));
				if (words.size() > 250)
				{
					workers.enqueue([&](size_t tId, const vector<string>& words)
					{
						for (auto& w : words) ++counters[tId][w];
					}, move(words));
				}
			}
			for (auto& w : words) ++counters[0][w];
		}
		counter = move(counters[0]);
		for (size_t i = 1; i < numWorkers - 1; ++i)
		{
			for (auto& p : counters[i]) counter[p.first] += p.second;
		}
	}
	else
	{
		for (size_t id = 0; ; ++id)
		{
			auto res = reader(id);
			if (res.stop) break;
			if (res.words.empty()) continue;
			minT = min(res.timePoint, minT);
			maxT = max(res.timePoint, maxT);
			for (auto& w : res.words) ++counter[w];
		}
	}
	zBias = minT;
	zSlope = minT == maxT ? 1 : 1 / (maxT - minT);
	for (auto& p : counter)
	{
		if (p.second < minCnt) continue;
		frequencies.emplace_back(p.second);
		vocabs.add(p.first);
	}
	buildModel();
}

size_t ChronoGramModel::recountVocab(const std::function<ReadResult(size_t)>& reader, float minT, float maxT, size_t numWorkers)
{
	if (!numWorkers) numWorkers = thread::hardware_concurrency();
	vector<vector<uint64_t>> counters(numWorkers);
	for (auto& c : counters) c.resize(vocabs.size());
	{
		ThreadPool workers(numWorkers, numWorkers * 16);
		vector<string> words;
		for (size_t id = 0; ; ++id)
		{
			auto res = reader(id);
			if (res.stop) break;
			if (res.words.empty()) continue;
			if (res.timePoint < minT) continue;
			if (res.timePoint > maxT) continue;
			words.insert(words.end(), make_move_iterator(res.words.begin()), make_move_iterator(res.words.end()));
			workers.enqueue([&](size_t tId, const vector<string>& words)
			{
				for (auto& w : words)
				{
					int id = vocabs.get(w);
					if(id >= 0) ++counters[tId][id];
				}
			}, move(words));
		}
	}

	frequencies = counters[0];
	for (size_t i = 1; i < numWorkers; ++i)
	{
		for (size_t n = 0; n < frequencies.size(); ++n) frequencies[n] += counters[i][n];
	}
	buildTable();
	return count_if(frequencies.begin(), frequencies.end(), [](auto e) { return e; });
}


size_t ChronoGramModel::recountVocab(const std::function<GNgramReadResult(size_t)>& reader, float minT, float maxT, size_t numWorkers)
{
	if (!numWorkers) numWorkers = thread::hardware_concurrency();
	fill(frequencies.begin(), frequencies.end(), 0);
	for (size_t id = 0; ; ++id)
	{
		auto res = reader(id);
		if (res.yearCnt.empty()) break;
		for (auto& yc : res.yearCnt)
		{
			if (yc.first < minT) continue;
			if (yc.first > maxT) continue;
			if (res.ngram[0] != (uint32_t)-1) frequencies[res.ngram[0]] += yc.second;
			if (res.ngram[4] != (uint32_t)-1) frequencies[res.ngram[4]] += yc.second;
		}
	}

	buildTable();
	return count_if(frequencies.begin(), frequencies.end(), [](auto e) { return e; });
}


bool ChronoGramModel::addFixedWord(const std::string & word)
{
	size_t wv = vocabs.get(word);
	if (wv == (size_t)-1) return false;
	fixedWords.emplace(wv);
	in.block(0, wv * L + 1, M, L - 1).setZero();
	return true;
}

void ChronoGramModel::train(const function<ReadResult(size_t)>& reader,
	size_t numWorkers, size_t windowLen, float fixedInit, float start_lr, size_t batch,
	size_t epoch, size_t report)
{
	if (!numWorkers) numWorkers = thread::hardware_concurrency();
	ThreadPool workers{ numWorkers };
	vector<ThreadLocalData> ld;
	if (numWorkers > 1)
	{
		ld.resize(numWorkers);
		for (auto& l : ld)
		{
			l.rg = mt19937_64{ globalData.rg() };
		}
	}
	vector<pair<vector<uint32_t>, float>> collections;
	vector<float> timePoints;
	uint64_t totW = accumulate(frequencies.begin(), frequencies.end(), 0ull);
	// estimate total size
	totalWords = totW * fixedInit;
	totalTimePoints = totalWords / 4;

	totalLL = totalLLCnt = 0;
	procWords = lastProcWords = 0;

	size_t read = 0;
	const auto& procCollection = [&](bool fixed)
	{
		if (collections.empty()) return;
		shuffle(collections.begin(), collections.end(), globalData.rg);
		for (auto& c : collections)
		{
			size_t tpCnt = (c.first.size() + 2) / 4;
			if(tpCnt) timePoints.resize(timePoints.size() + tpCnt, c.second);
		}
		shuffle(timePoints.begin(), timePoints.end(), globalData.rg);

		if (numWorkers > 1)
		{
			vector<future<void>> futures;
			futures.reserve(collections.size());
			for (auto& d : collections)
			{
				if (fixed)
				{
					futures.emplace_back(workers.enqueue([&](size_t threadId)
					{
						trainVectorsMulti<true>(d.first.data(), d.first.size(), d.second,
							windowLen, start_lr, epoch, report, ld[threadId]);
					}));
				}
				else
				{
					futures.emplace_back(workers.enqueue([&](size_t threadId)
					{
						trainVectorsMulti(d.first.data(), d.first.size(), d.second,
							windowLen, start_lr, epoch, report, ld[threadId]);
					}));
				}
			}
			if(!fixed && zeta > 0) trainTimePrior(timePoints.data(), timePoints.size(), start_lr, report);
			for (auto& f : futures) f.get();
			if (!fixed && zeta > 0)
			{
				normalizeWordDist(false);
			}
		}
		else
		{
			for (auto& d : collections)
			{
				if (fixed)
				{
					trainVectors<true>(d.first.data(), d.first.size(), d.second,
						windowLen, start_lr, epoch, report);
				}
				else
				{
					trainVectors(d.first.data(), d.first.size(), d.second,
						windowLen, start_lr, epoch, report);
				}
			}
			if (!fixed && zeta > 0)
			{
				trainTimePrior(timePoints.data(), timePoints.size(), start_lr, report);
				normalizeWordDist(false);
			}
		}
		collections.clear();
		timePoints.clear();
	};

	if (fixedInit > 0)
	{
		size_t flr = fixedInit;
		float frac = fmod(fixedInit, 1.f);
		for (size_t e = 0; e <= flr; ++e)
		{
			for (size_t id = 0; ; ++id)
			{
				if (procWords >= (e == flr ? frac : 1) * totW) break;
				auto rresult = reader(id);
				if (rresult.words.empty()) break;

				vector<uint32_t> doc;
				doc.reserve(rresult.words.size());
				for (auto& w : rresult.words)
				{
					auto id = vocabs.get(w);
					if (id < 0) continue;
					float ww = subsampling / (frequencies[id] / (float)totW);
					if (subsampling > 0 &&
						generate_canonical<float, 24>(globalData.rg) > sqrt(ww) + ww)
					{
						procWords++;
						continue;
					}
					doc.emplace_back(id);
				}

				if (doc.size() < 2)
				{
					procWords += doc.size();
					continue;
				}

				collections.emplace_back(make_pair(move(doc), rresult.timePoint));
				if (collections.size() >= batch)
				{
					procCollection(true);
				}
			}
			procCollection(true);
		}
		fprintf(stderr, "All vectors were initialized as fixed word embeddeings with ll %f\n", totalLL);
	}
	procWords = lastProcWords = 0;
	totalWords = totW * epoch;
	totalTimePoints = totalWords / 4;
	totalLL = ugLL = totalLLCnt = 0;
	timeLL = timeLLCnt = 0;
	procTimePoints = 0;
	timer.reset();

	for (size_t e = 0; e < epoch; ++e)
	{
		for (size_t id = 0; ; ++id)
		{
			auto rresult = reader(id);
			if (rresult.words.empty()) break;

			vector<uint32_t> doc;
			doc.reserve(rresult.words.size());
			for (auto& w : rresult.words)
			{
				auto id = vocabs.get(w);
				if (id < 0) continue;
				float ww = subsampling / (frequencies[id] / (float)totW);
				if (subsampling > 0 &&
					generate_canonical<float, 24>(globalData.rg) > sqrt(ww) + ww)
				{
					procWords++;
					continue;
				}
				doc.emplace_back(id);
			}

			if (doc.size() < 2)
			{
				procWords += doc.size();
				continue;
			}

			collections.emplace_back(make_pair(move(doc), rresult.timePoint));
			if (collections.size() >= batch)
			{
				procCollection(false);
			}
		}
	}
	procCollection(false);
	normalizeWordDist();
}

void ChronoGramModel::buildVocabFromDict(const function<pair<string, uint64_t>()>& reader, float minT, float maxT)
{
	zBias = minT;
	zSlope = minT == maxT ? 1 : 1 / (maxT - minT);
	pair<string, uint64_t> p;
	while((p = reader()).second > 0)
	{
		frequencies.emplace_back(p.second);
		vocabs.add(p.first);
	}
	buildModel();
}

void ChronoGramModel::trainFromGNgram(const function<GNgramReadResult(size_t)>& reader, uint64_t maxItems,
	size_t numWorkers, float fixedInit, float start_lr, size_t batchSents, size_t epochs, size_t report)
{
	if (!numWorkers) numWorkers = thread::hardware_concurrency();
	ThreadPool workers{ numWorkers };
	vector<ThreadLocalData> ld;
	if (numWorkers > 1)
	{
		ld.resize(numWorkers);
		for (auto& l : ld)
		{
			l.rg = mt19937_64{ globalData.rg() };
		}
	}

	vector<array<uint32_t, 5>> ngrams;
	vector<pair<uint32_t, float>> collections;
	vector<float> timePoints;
	
	const float minT = getMinPoint(), maxT = getMaxPoint();

	uint64_t totW = accumulate(frequencies.begin(), frequencies.end(), 0ull);
	totalWords = maxItems * fixedInit;
	totalTimePoints = totalWords / 8;

	totalLL = totalLLCnt = 0;
	procWords = lastProcWords = 0;

	size_t read = 0;
	const auto& procCollection = [&](bool fixed)
	{
		if (collections.empty()) return;
		shuffle(collections.begin(), collections.end(), globalData.rg);

		timePoints.resize(collections.size() / 8);
		transform(collections.begin(), collections.begin() + timePoints.size(), timePoints.begin(), [](auto p) { return p.second; });

		if (numWorkers > 1)
		{
			vector<future<void>> futures;
			futures.reserve(collections.size());
			for (auto& d : collections)
			{
				if (fixed)
				{
					futures.emplace_back(workers.enqueue([&](size_t threadId)
					{
						trainVectorsMulti<true, true>(ngrams[d.first].data(), 5, d.second,
							4, start_lr, epochs, report, ld[threadId]);
					}));
				}
				else
				{
					futures.emplace_back(workers.enqueue([&](size_t threadId)
					{
						trainVectorsMulti<false, true>(ngrams[d.first].data(), 5, d.second,
							4, start_lr, epochs, report, ld[threadId]);
					}));
				}
			}
			if (!fixed && zeta > 0) trainTimePrior(timePoints.data(), timePoints.size(), start_lr, report);
			for (auto& f : futures) f.get();
			if (!fixed && zeta > 0)
			{
				normalizeWordDist(false);
			}
		}
		else
		{
			for (auto& d : collections)
			{
				if (fixed)
				{
					trainVectors<true, true>(ngrams[d.first].data(), 5, d.second,
						4, start_lr, epochs, report);
				}
				else
				{
					trainVectors<false, true>(ngrams[d.first].data(), 5, d.second,
						4, start_lr, epochs, report);
				}
			}
			if (!fixed && zeta > 0)
			{
				trainTimePrior(timePoints.data(), timePoints.size(), start_lr, report);
				normalizeWordDist(false);
			}
		}
		collections.clear();
		ngrams.clear();
		timePoints.clear();

		float time_per_kword = (procWords - lastProcWords) / timer.getElapsed() / 1000.f;
		fprintf(stderr, "%.2f%% %.4f %.4f %.4f %.2f kwords/sec\n",
			procWords / (totalWords / 100.f), totalLL + ugLL, totalLL, ugLL, time_per_kword);
		lastProcWords = procWords;
		timer.reset();
	};

	if (fixedInit > 0)
	{
		size_t flr = fixedInit;
		float frac = fmod(fixedInit, 1.f);
		for (size_t e = 0; e <= flr; ++e)
		{
			for (size_t id = 0; id < maxItems * (e == flr ? frac : 1); ++id)
			{
				auto rresult = reader(id);
				if (rresult.yearCnt.empty()) break;

				for (auto& w : rresult.ngram)
				{
					if (w == (uint32_t)-1) continue;
					float ww = subsampling / (frequencies[w] / (float)totW);
					if (subsampling > 0 &&
						generate_canonical<float, 24>(globalData.rg) > sqrt(ww) + ww)
					{
						w = -1;
					}
				}

				++procWords;
				if ((rresult.ngram[0] == (uint32_t)-1 && rresult.ngram[4] == (uint32_t)-1) ||
					count(rresult.ngram.begin(), rresult.ngram.end(), (uint32_t)-1) == 4)
				{
					continue;
				}

				for (auto& p : rresult.yearCnt)
				{
					if (p.first < minT || p.first > maxT) continue;
					for (size_t cnt = 0; cnt < p.second; ++cnt)
					{
						collections.emplace_back(ngrams.size(), p.first);
					}
				}

				ngrams.emplace_back(rresult.ngram);

				if (collections.size() >= batchSents)
				{
					procCollection(true);
				}
			}
			procCollection(true);
		}
		fprintf(stderr, "All vectors were initialized as fixed word embeddeings with ll %f\n", totalLL);
	}
	procWords = lastProcWords = 0;
	totalWords = maxItems * epochs;
	totalTimePoints = totalWords / 8;
	totalLL = totalLLCnt = 0;
	timeLL = timeLLCnt = 0;
	procTimePoints = 0;
	timer.reset();

	for (size_t e = 0; e < epochs; ++e)
	{
		for (size_t id = 0; id < maxItems; ++id)
		{
			auto rresult = reader(id);
			if (rresult.yearCnt.empty()) break;

			for (auto& w : rresult.ngram)
			{
				if (w == (uint32_t)-1) continue;
				float ww = subsampling / frequencies[w] * totW;
				if (subsampling > 0 &&
					generate_canonical<float, 24>(globalData.rg) > sqrt(ww) + ww)
				{
					w = -1;
				}
			}

			++procWords;
			if ((rresult.ngram[0] == (uint32_t)-1 && rresult.ngram[4] == (uint32_t)-1) || 
				count(rresult.ngram.begin(), rresult.ngram.end(), (uint32_t)-1) == 4)
			{
				continue;
			}

			for (auto& p : rresult.yearCnt)
			{
				if (p.first < minT || p.first > maxT) continue;
				for (size_t cnt = 0; cnt < p.second; ++cnt)
				{
					collections.emplace_back(ngrams.size(), p.first);
				}
			}

			ngrams.emplace_back(rresult.ngram);

			if (collections.size() >= batchSents)
			{
				procCollection(false);
			}
		}
	}
	procCollection(false);
	normalizeWordDist();
}

float ChronoGramModel::arcLengthOfWord(const string & word, size_t step) const
{
	size_t wv = vocabs.get(word);
	if (wv == (size_t)-1) return {};
	float len = 0;
	VectorXf v = makeTimedVector(wv, makeCoef(L, timePadding));
	for (size_t i = 0; i < step; ++i)
	{
		VectorXf u = makeTimedVector(wv, makeCoef(L, (float)(i + 1) / step * (1 - timePadding * 2) + timePadding));
		len += (v - u).norm();
		v.swap(u);
	}
	return len;
}

float ChronoGramModel::angleOfWord(const std::string & word, size_t step) const
{
	size_t wv = vocabs.get(word);
	if (wv == (size_t)-1) return {};
	float angle = 0;
	VectorXf v = makeTimedVector(wv, makeCoef(L, timePadding));
	for (size_t i = 0; i < step; ++i)
	{
		VectorXf u = makeTimedVector(wv, makeCoef(L, (float)(i + 1) / step * (1 - timePadding * 2) + timePadding));
		angle += acos(u.normalized().dot(v.normalized()));
		v.swap(u);
	}
	return angle;
}

vector<tuple<string, float, float>> ChronoGramModel::nearestNeighbors(const string & word,
	float wordTimePoint, float searchingTimePoint, float m, size_t K) const
{
	return mostSimilar({ make_pair(word, wordTimePoint) }, {}, searchingTimePoint, m, K);
}

vector<tuple<string, float, float>> ChronoGramModel::mostSimilar(
	const vector<pair<string, float>>& positiveWords,
	const vector<pair<string, float>>& negativeWords,
	float searchingTimePoint, float m, size_t K) const
{
	constexpr float threshold = 0.4f;
	constexpr float bias = 0.2f;
	VectorXf vec = VectorXf::Zero(M);
	const size_t V = vocabs.size();
	unordered_set<size_t> uniqs;
	VectorXf coef = makeCoef(L, normalizedTimePoint(searchingTimePoint));
	for (auto& p : positiveWords)
	{
		size_t wv = vocabs.get(p.first);
		if (wv == (size_t)-1) return {};
		VectorXf cf = makeCoef(L, normalizedTimePoint(p.second));
		float tPrior = getTimePrior(cf);
		VectorXf tv = makeTimedVector(wv, cf);
		float wPrior = getWordProbByTime(wv, tv, tPrior);
		if (wPrior / (tPrior + bias) < threshold) continue;
		vec += tv * (1 - m) + out.col(wv) * m;
		uniqs.emplace(wv);
	}

	for (auto& p : negativeWords)
	{
		size_t wv = vocabs.get(p.first);
		if (wv == (size_t)-1) return {};
		VectorXf cf = makeCoef(L, normalizedTimePoint(p.second));
		float tPrior = getTimePrior(cf);
		VectorXf tv = makeTimedVector(wv, cf);
		float wPrior = getWordProbByTime(wv, tv, tPrior);
		if (wPrior / (tPrior + bias) < threshold) continue;
		vec -= tv * (1 - m) + out.col(wv) * m;
		uniqs.emplace(wv);
	}

	vec.normalize();
	float tPrior = getTimePrior(coef);
	vector<tuple<string, float, float>> top;
	vector<pair<float, float>> sim(V);
	for (size_t v = 0; v < V; ++v)
	{
		if (!frequencies[v])
		{
			sim[v] = make_pair(-INFINITY, 0);
			continue;
		}
		VectorXf tv = makeTimedVector(v, coef);
		float wPrior = getWordProbByTime(v, tv, tPrior);
		if (wPrior / (tPrior + bias) < threshold)
		{
			sim[v] = make_pair(-INFINITY, wPrior / tPrior);
			continue;
		}
		sim[v] = make_pair((tv * (1 - m) + out.col(v) * m).normalized().dot(vec), wPrior / tPrior);
	}

	for (size_t k = 0; k < K; ++k)
	{
		size_t idx = max_element(sim.begin(), sim.end()) - sim.begin();
		top.emplace_back(vocabs.getStr(idx), sim[idx].first, sim[idx].second);
		sim.data()[idx].first = -INFINITY;
	}
	return top;
}

vector<tuple<string, float>> ChronoGramModel::mostSimilarStatic(
	const vector<string>& positiveWords,
	const vector<string>& negativeWords,
	size_t K) const
{
	VectorXf vec = VectorXf::Zero(M);
	const size_t V = vocabs.size();
	unordered_set<size_t> uniqs;
	for (auto& p : positiveWords)
	{
		size_t wv = vocabs.get(p);
		if (wv == (size_t)-1) return {};
		vec += out.col(wv);
		uniqs.emplace(wv);
	}

	for (auto& p : negativeWords)
	{
		size_t wv = vocabs.get(p);
		if (wv == (size_t)-1) return {};
		vec -= out.col(wv);
		uniqs.emplace(wv);
	}

	vec.normalize();

	vector<tuple<string, float>> top;
	VectorXf sim(V);
	for (size_t v = 0; v < V; ++v)
	{
		if (uniqs.count(v) || !frequencies[v])
		{
			sim(v) = -INFINITY;
			continue;
		}
		sim(v) = out.col(v).normalized().dot(vec);
	}

	for (size_t k = 0; k < K; ++k)
	{
		size_t idx = max_element(sim.data(), sim.data() + sim.size()) - sim.data();
		top.emplace_back(vocabs.getStr(idx), sim.data()[idx]);
		sim.data()[idx] = -INFINITY;
	}
	return top;
}

vector<pair<string, float>> ChronoGramModel::calcShift(size_t minCnt, float time1, float time2, float m) const
{
	VectorXf coef1 = makeCoef(L, normalizedTimePoint(time1)),
		coef2 = makeCoef(L, normalizedTimePoint(time2));
	vector<pair<string, float>> ret;
	const size_t V = vocabs.size();
	float tPrior1 = getTimePrior(coef1);
	float tPrior2 = getTimePrior(coef2);
	constexpr float threshold = 0.4f;
	constexpr float bias = 0.2f;

	for (size_t v = 0; v < V; ++v)
	{
		if (frequencies[v] < minCnt) continue;
		VectorXf v1 = makeTimedVector(v, coef1);
		VectorXf v2 = makeTimedVector(v, coef2);
		if (getWordProbByTime(v, v1, tPrior1) / (tPrior1 + bias) < threshold) continue;
		if (getWordProbByTime(v, v2, tPrior2) / (tPrior2 + bias) < threshold) continue;
		float d;
		if (m >= 0)
		{
			d = 1 - (v1 * (1 - m) + out.col(v) * m).normalized().dot((v2 * (1 - m) + out.col(v) * m).normalized());
		}
		else
		{
			d = (v1 - v2).norm();
		}
		ret.emplace_back(vocabs.getStr(v), d);
	}
	
	sort(ret.begin(), ret.end(), [](auto p1, auto p2)
	{
		return p1.second < p2.second;
	});
	return ret;
}

float ChronoGramModel::sumSimilarity(const string & src, const vector<string>& targets, float timePoint, float m) const
{
	constexpr float threshold = 0.4f;
	constexpr float bias = 0.2f;
	const size_t V = vocabs.size();
	unordered_set<size_t> uniqs;
	VectorXf coef = makeCoef(L, normalizedTimePoint(timePoint));
	size_t wv = vocabs.get(src);
	if (wv == (size_t)-1) return -INFINITY;
	float tPrior = getTimePrior(coef);
	VectorXf tv = makeTimedVector(wv, coef);
	float wPrior = getWordProbByTime(wv, tv, tPrior);
	if (wPrior / (tPrior + bias) < threshold) return -INFINITY;
	VectorXf vec = (tv * (1 - m) + out.col(wv) * m).normalized();

	vector<size_t> targetWv;
	for (auto& w : targets)
	{
		size_t wv = vocabs.get(w);
		if (wv == (size_t)-1) continue;
		targetWv.emplace_back(wv);
	}

	float sum = 0;
	for (auto v : targetWv)
	{
		VectorXf tv = makeTimedVector(v, coef);
		float wPrior = getWordProbByTime(v, tv, tPrior);
		if (wPrior / (tPrior + bias) < threshold) continue;
		sum += (tv * (1 - m) + out.col(v) * m).normalized().dot(vec);
	}

	return sum;
}

float ChronoGramModel::similarity(const string & word1, float time1, const string & word2, float time2) const
{
	size_t wv1 = vocabs.get(word1), wv2 = vocabs.get(word2);
	if (wv1 == (size_t)-1 || wv2 == (size_t)-1) return 0;
	VectorXf c1 = makeCoef(L, normalizedTimePoint(time1)), c2 = makeCoef(L, normalizedTimePoint(time2));
	return makeTimedVector(wv1, c1).normalized().dot(makeTimedVector(wv2, c2).normalized());
}

float ChronoGramModel::similarity(const string & word1, const string & word2) const
{
	size_t wv1 = vocabs.get(word1), wv2 = vocabs.get(word2);
	if (wv1 == (size_t)-1 || wv2 == (size_t)-1) return 0;
	return out.col(wv1).normalized().dot(out.col(wv2).normalized());
}


ChronoGramModel::LLEvaluater ChronoGramModel::evaluateSent(const vector<string>& words, 
	size_t windowLen, size_t nsQ, const function<float(float)>& timePrior, float timePriorWeight) const
{
	const size_t V = vocabs.size();
	vector<uint32_t> wordIds;
	unordered_set<uint32_t> uniqs;
	unordered_map<uint32_t, LLEvaluater::MixedVectorCoef> coefs;
	transform(words.begin(), words.end(), back_inserter(wordIds), [&](auto w) { return vocabs.get(w); });
	uniqs.insert(wordIds.begin(), wordIds.end());
	size_t n = 0;
	for (auto i : wordIds)
	{
		if (i == (uint32_t)-1) continue;
		if (coefs.count(i)) continue;
		coefs[i].n = n;
		if (nsQ) coefs[i].dataVec.resize((V + nsQ - 1) / nsQ * L);
		for (size_t v = 0; v < V; ++v)
		{
			if (uniqs.count(v) == 0 && (!nsQ || v % nsQ != n)) continue;
			VectorXf c = out.col(v).transpose() * in.block(0, i * L, M, L);
			if (nsQ && v % nsQ == n)
			{
				copy(c.data(), c.data() + L, &coefs[i].dataVec[v / nsQ * L]);
			}
			else
			{
				coefs[i].dataMap[v] = { c.data(), c.data() + L };
			}
		}
		if (nsQ) n = (n + 1) % nsQ;
	}

	return LLEvaluater(*this, windowLen, nsQ, move(wordIds), move(coefs), timePrior, timePriorWeight);
}


pair<float, float> ChronoGramModel::predictSentTime(const vector<string>& words, 
	size_t windowLen, size_t nsQ, const function<float(float)>& timePrior, float timePriorWeight,
	size_t initStep, float threshold, std::vector<float>* llOutput) const
{
	auto evaluator = evaluateSent(words, windowLen, nsQ, timePrior, timePriorWeight);
	constexpr uint32_t SCALE = 0x80000000;
	map<uint32_t, pair<float, float>> lls;
	float maxLL = -INFINITY;
	uint32_t maxP = 0;
	if (llOutput) llOutput->clear();
	for (size_t i = 0; i <= initStep; ++i)
	{
		auto t = evaluator.fg(i / (float)initStep * (1 - timePadding * 2) + timePadding);
		auto m = (uint32_t)(SCALE * (float)i / initStep);
		lls[m] = make_pair(get<0>(t), get<1>(t));
		if (get<0>(t) > maxLL)
		{
			maxP = m;
			maxLL = get<0>(t);
		}
		if (llOutput) llOutput->emplace_back(get<0>(t));
	}

	for (auto it = ++lls.begin(); it != lls.end(); ++it)
	{
		auto prevIt = prev(it);
		if (it->first - prevIt->first < (uint32_t)(SCALE * threshold)) continue;
		if (prevIt->second.second < 0) continue;
		if (it->second.second > 0) continue;
		auto m = (prevIt->first + it->first) / 2;
		auto t = evaluator.fg(m / (float)SCALE * (1 - timePadding * 2) + timePadding);
		lls.emplace(m, make_pair(get<0>(t), get<1>(t)));
		it = prevIt;
		if (get<0>(t) > maxLL)
		{
			maxP = m;
			maxLL = get<0>(t);
		}
	}

	return make_pair(unnormalizedTimePoint(maxP / (float)SCALE * (1 - timePadding * 2) + timePadding), maxLL);
}

vector<ChronoGramModel::EvalResult> ChronoGramModel::evaluate(const function<ReadResult(size_t)>& reader,
	const function<void(EvalResult)>& writer, size_t numWorkers, 
	size_t windowLen, size_t nsQ, const function<float(float)>& timePrior, float timePriorWeight,
	size_t initStep, float threshold) const
{
	if (!numWorkers) numWorkers = thread::hardware_concurrency();
	vector<EvalResult> ret;
	ThreadPool workers{ numWorkers };
	map<size_t, EvalResult> res;
	size_t outputId = 0;
	mutex readMtx, writeMtx;
	condition_variable readCnd;
	size_t id = 0;
	auto consume = [&]()
	{
		lock_guard<mutex> l{ writeMtx };
		while (!res.empty() && res.begin()->first == outputId)
		{
			if (writer) writer(move(res.begin()->second));
			else ret.emplace_back(move(res.begin()->second));
			res.erase(res.begin());
			outputId++;
		}
	};

	for (;; ++id)
	{
		auto r = reader(id);
		if (r.stop) break;
		unique_lock<mutex> l{ readMtx };
		readCnd.wait(l, [&]() { return workers.getNumEnqued() < workers.getNumWorkers() * 4; });
		consume();
		workers.enqueue([&, id](size_t tid, float time, vector<string> words)
		{
			vector<float> lls;
			auto p = predictSentTime(words, windowLen, nsQ, timePrior, timePriorWeight, initStep, threshold, &lls);
			lock_guard<mutex> l{ writeMtx };
			res[id] = { time, p.first, p.second, p.second / 2 / windowLen / words.size(), 
				(p.first - time) * zSlope, move(words), move(lls) };
			readCnd.notify_all();
		}, r.timePoint, move(r.words));
	}
	while (outputId < id)
	{
		unique_lock<mutex> l{ readMtx };
		readCnd.wait(l, [&]() { return !res.empty() || outputId >= id; });
		consume();
	}
	return ret;
}

MatrixXf ChronoGramModel::getEmbeddingMatrix(const string & word) const
{
	size_t wv = vocabs.get(word);
	if (wv == (size_t)-1) return {};
	return in.block(0, wv * L, M, L);
}

VectorXf ChronoGramModel::getEmbedding(const string & word) const
{
	size_t wv = vocabs.get(word);
	if (wv == (size_t)-1) return {};
	return out.col(wv);
}

VectorXf ChronoGramModel::getEmbedding(const string & word, float timePoint) const
{
	size_t wv = vocabs.get(word);
	if (wv == (size_t)-1) return {};
	return makeTimedVector(wv, makeCoef(L, normalizedTimePoint(timePoint)));
}

void ChronoGramModel::saveModel(ostream & os, bool compressed) const
{
	os.write("CHGR", 4);
	writeToBinStream(os, (uint32_t)(compressed ? 2 : 1));
	writeToBinStream(os, (uint32_t)M);
	writeToBinStream(os, (uint32_t)L);
	writeToBinStream(os, zBias);
	writeToBinStream(os, zSlope);
	vocabs.writeToFile(os);
	writeToBinStream(os, frequencies);
	if (compressed)
	{
		writeToBinStreamCompressed(os, in);
		writeToBinStreamCompressed(os, out);
	}
	else
	{
		writeToBinStream(os, in);
		writeToBinStream(os, out);
	}
	writeToBinStream(os, zeta);
	writeToBinStream(os, lambda);
	writeToBinStream(os, timePadding);
	writeToBinStream(os, timePrior);
}

ChronoGramModel ChronoGramModel::loadModel(istream & is)
{
	auto pos = is.tellg();
	char buf[5] = { 0, };
	is.read(buf, 4);
	if (strcmp(buf, "CHGR") == 0)
	{
		size_t version = readFromBinStream<uint32_t>(is);
		size_t M = readFromBinStream<uint32_t>(is);
		size_t L = readFromBinStream<uint32_t>(is);
		ChronoGramModel ret{ M, L };
		ret.zBias = readFromBinStream<float>(is);
		ret.zSlope = readFromBinStream<float>(is);
		ret.vocabs.readFromFile(is);
		size_t V = ret.vocabs.size();
		ret.in.resize(M, L * V);
		ret.out.resize(M, V);

		readFromBinStream(is, ret.frequencies);
		if (version == 1)
		{
			readFromBinStream(is, ret.in);
			readFromBinStream(is, ret.out);
		}
		else
		{
			readFromBinStreamCompressed(is, ret.in);
			readFromBinStreamCompressed(is, ret.out);
		}
		readFromBinStream(is, ret.zeta);
		readFromBinStream(is, ret.lambda);
		readFromBinStream(is, ret.timePadding);
		ret.timePrior.resize(L);
		readFromBinStream(is, ret.timePrior);
		ret.buildTable();
		ret.normalizeWordDist();
		return ret;
	}
	else
	{
		is.seekg(pos);
		size_t M = readFromBinStream<uint32_t>(is);
		size_t L = readFromBinStream<uint32_t>(is);
		ChronoGramModel ret{ M, L };
		ret.zBias = readFromBinStream<float>(is);
		ret.zSlope = readFromBinStream<float>(is);
		ret.vocabs.readFromFile(is);
		size_t V = ret.vocabs.size();
		ret.in.resize(M, L * V);
		ret.out.resize(M, V);

		readFromBinStream(is, ret.frequencies);
		readFromBinStream(is, ret.in);
		readFromBinStream(is, ret.out);

		try
		{
			readFromBinStream(is, ret.zeta);
			readFromBinStream(is, ret.lambda);
			readFromBinStream(is, ret.timePadding);
			ret.timePrior.resize(L);
			readFromBinStream(is, ret.timePrior);
		}
		catch (const exception& e)
		{
			ret.timePadding = 0;
			ret.timePrior = VectorXf::Zero(L);
			ret.timePrior[0] = 1;
		}
		ret.buildTable();
		ret.normalizeWordDist();
		return ret;
	}
}

float ChronoGramModel::getWordProbByTime(const std::string & word, float timePoint) const
{
	size_t wv = vocabs.get(word);
	if (wv == (size_t)-1) return 0;
	return getWordProbByTime(wv, timePoint);
}

float ChronoGramModel::getTimePrior(float timePoint) const
{
	return getTimePrior(makeCoef(L, normalizedTimePoint(timePoint)));
}

size_t ChronoGramModel::getTotalWords() const
{
	return accumulate(frequencies.begin(), frequencies.end(), 0);
}

size_t ChronoGramModel::getWordCount(const std::string & word) const
{
	size_t wv = vocabs.get(word);
	if (wv == (size_t)-1) return 0;
	return frequencies[wv];
}

float ChronoGramModel::LLEvaluater::operator()(float timePoint) const
{
	const size_t N = wordIds.size(), V = tgm.unigramDist.size();
	auto tCoef = makeCoef(tgm.L, timePoint);
	auto defaultPrior = [&](float)->float
	{
		return log(1 - exp(-pow(tgm.timePrior.dot(tCoef), 2) / 2) + 1e-5f);
	};
	float pa = max(tgm.getTimePrior(tCoef), 1e-5f);
	float ll = (timePrior ? timePrior : defaultPrior)(tgm.unnormalizedTimePoint(timePoint)) * timePriorWeight;
	unordered_map<uint32_t, uint32_t> count;

	for (size_t i = 0; i < N; ++i)
	{
		const uint32_t x = wordIds[i];
		if (x == (uint32_t)-1) continue;
		auto& cx = coefs.find(x)->second;
		size_t jBegin = 0, jEnd = N - 1;
		if (i > windowLen) jBegin = i - windowLen;
		if (i + windowLen < N) jEnd = i + windowLen;

		for (size_t j = jBegin; j <= jEnd; ++j)
		{
			if (i == j) continue;
			const uint32_t y = wordIds[j];
			if (y == (uint32_t)-1) continue;
			float d = inner_product(tCoef.data(), tCoef.data() + tgm.L, cx.get(y, nsQ, tgm.L), 0.f);
			ll += logsigmoid(d) * (1 - tgm.zeta);
			count[x]++;
		}

		ll += log(1 - exp(-tgm.makeTimedVector(x, tCoef).squaredNorm() / 2 * tgm.lambda * pa) + 1e-5f) * tgm.zeta;
	}

	if (nsQ)
	{
		for (auto& p : count)
		{
			auto& cx = coefs.find(p.first)->second;
			float nll = 0;
			float denom = 0;
			for (size_t j = cx.n; j < V; j += nsQ)
			{
				float d = inner_product(tCoef.data(), tCoef.data() + tgm.L, cx.get(j, nsQ, tgm.L), 0.f);
				nll += tgm.unigramDist[j] * logsigmoid(-d);
				denom += tgm.unigramDist[j];
			}
			ll += nll / denom * tgm.negativeSampleSize * p.second * (1 - tgm.zeta);
		}
	}
	return ll;
}

tuple<float, float> ChronoGramModel::LLEvaluater::fg(float timePoint) const
{
	const size_t N = wordIds.size(), V = tgm.unigramDist.size();
	auto tCoef = makeCoef(tgm.L, timePoint), tDCoef = makeDCoef(tgm.L, timePoint);
	auto defaultPrior = [&](float)->float
	{
		return log(1 - exp(-pow(tgm.timePrior.dot(tCoef), 2) / 2) + 1e-5f);
	};
	float pa = max(tgm.getTimePrior(tCoef), 1e-5f);
	float dot = tgm.timePrior.dot(tCoef);
	float ddot = tgm.timePrior.block(1, 0, tgm.L - 1, 1).dot(tDCoef);
	float ll = (timePrior ? timePrior : defaultPrior)(tgm.unnormalizedTimePoint(timePoint)) * timePriorWeight,
		dll = (dot * ddot / (exp(pow(dot, 2)) - 1 + 1e-5f) + dot * ddot) * timePriorWeight;
	unordered_map<uint32_t, uint32_t> count;

	for (size_t i = 0; i < N; ++i)
	{
		const uint32_t x = wordIds[i];
		if (x == (uint32_t)-1) continue;
		auto& cx = coefs.find(x)->second;
		size_t jBegin = 0, jEnd = N - 1;
		if (i > windowLen) jBegin = i - windowLen;
		if (i + windowLen < N) jEnd = i + windowLen;

		for (size_t j = jBegin; j <= jEnd; ++j)
		{
			if (i == j) continue;
			const uint32_t y = wordIds[j];
			if (y == (uint32_t)-1) continue;
			float d = inner_product(tCoef.data(), tCoef.data() + tgm.L, cx.get(y, nsQ, tgm.L), 0.f);
			ll += logsigmoid(d) * (1 - tgm.zeta);
			float dd = inner_product(tDCoef.data(), tDCoef.data() + tgm.L - 1, cx.get(y, nsQ, tgm.L) + 1, 0.f), sd;
			dll += dd * (sd = sigmoid(-d)) * (1 - tgm.zeta);
			count[x]++;
		}

		auto v = tgm.makeTimedVector(x, tCoef);
		float sqn = v.squaredNorm() / 2 * tgm.lambda * pa, sd;
		ll += log(1 - exp(-sqn) + 1e-5f) * tgm.zeta;
		dll += v.dot(tgm.in.block(0, x * tgm.L + 1, tgm.M, tgm.L - 1) * tDCoef) * tgm.lambda * pa / (sd = exp(sqn) - 1 + 1e-5f) * tgm.zeta;
	}

	if (nsQ)
	{
		for (auto& p : count)
		{
			auto& cx = coefs.find(p.first)->second;
			float nll = 0, dnll = 0, ddnll = 0;
			float denom = 0;
			for (size_t j = cx.n; j < V; j += nsQ)
			{
				float d = inner_product(tCoef.data(), tCoef.data() + tgm.L, cx.get(j, nsQ, tgm.L), 0.f);
				nll += tgm.unigramDist[j] * logsigmoid(-d);
				float dd = inner_product(tDCoef.data(), tDCoef.data() + tgm.L - 1, cx.get(j, nsQ, tgm.L) + 1, 0.f), sd;
				dnll += -tgm.unigramDist[j] * dd * (sd = sigmoid(d));
				denom += tgm.unigramDist[j];
			}
			ll += nll / denom * tgm.negativeSampleSize * p.second * (1 - tgm.zeta);
			dll += dnll / denom * tgm.negativeSampleSize * p.second * (1 - tgm.zeta);
		}
	}
	return make_tuple(ll, dll);
}
