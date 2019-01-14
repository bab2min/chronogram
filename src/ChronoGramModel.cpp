#include <numeric>
#include <iostream>
#include <iterator>
#include <unordered_set>
#include "ChronoGramModel.h"
#include "IOUtils.h"
#include "ThreadPool.h"
#include "polynomials.hpp"

using namespace std;
using namespace Eigen;

void ChronoGramModel::buildModel()
{
	const size_t V = vocabs.size();
	// allocate & initialize model
	in = MatrixXf::Random(M, L * V) * (.5f / M);
	out = MatrixXf::Random(M, V) * (.5f / M);

	timePrior = VectorXf::Zero(L);
	timePrior[0] = 1;
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

float ChronoGramModel::avgTimeSqNorm(size_t wv) const
{
	float ret = 0;
	for (size_t l = 0; l < L; ++l)
	{
		ret += in.col(wv * L + l).squaredNorm() * integratedSqChebyshevT(l);
		for (size_t m = 0; m < l; ++m)
		{
			ret += 2 * in.col(wv * L + l).dot(in.col(wv * L + m)) * integratedChebyshevTT(l, m);
		}
	}
	return ret;
}

float ChronoGramModel::avgTimePrior() const
{
	float ret = 0;
	for (size_t l = 0; l < L; ++l)
	{
		ret += pow(timePrior[l], 2) * integratedSqChebyshevT(l);
		for (size_t m = 0; m < l; ++m)
		{
			ret += 2 * timePrior[l] * timePrior[m] * integratedChebyshevTT(l, m);
		}
	}
	return ret;
}

float ChronoGramModel::inplaceUpdate(size_t x, size_t y, float lr, bool negative, const VectorXf& lWeight)
{
	auto outcol = out.col(y);
	VectorXf inSum = makeTimedVector(x, lWeight);
	float f = inSum.dot(outcol);
	float pr = logsigmoid(f * (negative ? -1 : 1));

	float d = (negative ? 0 : 1) - sigmoid(f);
	float g = lr * d;

	VectorXf in_grad = g * outcol;
	VectorXf out_grad = g * inSum;
	outcol += out_grad;

	if (fixedWords.count(x))
	{
		in.col(x * L) += in_grad;
	}
	else
	{
		in.block(0, x*L, M, L) += in_grad * VectorXf{ lWeight.array() * vEta.array() }.transpose();
	}
	return pr;
}

float ChronoGramModel::getUpdateGradient(size_t x, size_t y, float lr, bool negative, const VectorXf& lWeight,
	Eigen::DenseBase<Eigen::MatrixXf>::ColXpr xGrad, Eigen::DenseBase<Eigen::MatrixXf>::ColXpr yGrad)
{
	auto outcol = out.col(y);
	VectorXf inSum = makeTimedVector(x, lWeight);
	float f = inSum.dot(outcol);
	float pr = logsigmoid(f * (negative ? -1 : 1));

	float d = (negative ? 0 : 1) - sigmoid(f);
	float g = lr * d;

	xGrad += g * outcol;
	yGrad += g * inSum;
	return max(pr, -100.f);
}

float ChronoGramModel::inplaceTimeUpdate(size_t x, float lr, const VectorXf& lWeight)
{
	/*
	log P(x|t) = log(1 - exp(-x^2/2 * l))
	d log P(x|t) / dx = lx / (exp(x^2/2 * l) - 1)

	log (1-P(x|t)) = log(exp(-x^2/2 * l)) = -x^2/2 * l
	d log (1-P(x|t)) / dx = -x * l
	*/
	auto gPos = makeTimedVector(x, lWeight);
	float pr = log(1 - exp(-gPos.squaredNorm() / 2 * lambda) + 1e-5);
	gPos /= exp(gPos.squaredNorm() / 2 * lambda) - 1 + 1e-3;
	pr += -avgTimeSqNorm(x) * lambda;

	in.block(0, x * L, M, L) += gPos * lWeight.transpose() * lambda * lr;
	in.block(0, x * L, M, L) += -in.block(0, x * L, M, L) * avgNegMatrix * lambda * lr;
	return pr;
}

float ChronoGramModel::getTimeUpdateGradient(size_t x, float lr, const Eigen::VectorXf & lWeight, Eigen::Block<Eigen::MatrixXf> grad)
{
	/*
	log P(x|t) = log(1 - exp(-x^2/2 * l))
	d log P(x|t) / dx = lx / (exp(x^2/2 * l) - 1)

	log (1-P(x|t)) = log(exp(-x^2/2 * l)) = -x^2/2 * l
	d log (1-P(x|t)) / dx = -x * l
	*/
	auto gPos = makeTimedVector(x, lWeight);
	float pr = log(1 - exp(-gPos.squaredNorm() / 2 * lambda) + 1e-5);
	gPos /= exp(gPos.squaredNorm() / 2 * lambda) - 1 + 1e-3;
	pr += -avgTimeSqNorm(x) * lambda;

	grad += gPos * lWeight.transpose() * lambda * lr;
	grad += -in.block(0, x * L, M, L) * avgNegMatrix * lambda * lr;
	return pr;
}


float ChronoGramModel::updateTimePrior(float lr, const Eigen::VectorXf & lWeight)
{
	float p = timePrior.dot(lWeight);
	float pr = log(1 - exp(-p * p / 2) + 1e-5);
	p /= exp(p * p / 2) - 1 + 1e-3;
	pr += -avgTimePrior();
	timePrior += p * lWeight * lr;
	timePrior -= avgNegMatrix * timePrior * lr;
	return pr;
}


void ChronoGramModel::trainVectors(const uint32_t * ws, size_t N, float timePoint,
	size_t window_length, float start_lr, size_t nEpoch, size_t report)
{
	uniform_int_distribution<size_t> uid{ 0, window_length > 2 ? window_length - 2 : 0 };
	VectorXf coef = makeCoef(L, normalizedTimePoint(timePoint));

	for (size_t i = 0; i < N; ++i)
	{
		const auto& x = ws[i];
		float lr1 = max(start_lr * (1 - procWords / (totalWords + 1.f)), start_lr * 1e-4f);
		size_t jBegin = 0, jEnd = N;
		if (i > window_length) jBegin = i - window_length;
		if (i + window_length < N) jEnd = i + window_length;

		float llSum = 0;
		size_t llCnt = 0;
		// update in, out vector
		for (auto j = jBegin; j < jEnd; ++j)
		{
			if (i == j) continue;
			float ll = inplaceUpdate(x, ws[j], lr1 * (1 - zeta), false, coef);
			assert(isnormal(ll));
			for (size_t k = 0; k < negativeSampleSize; ++k)
			{
				uint32_t ns = unigramTable(globalData.rg);
				while (ns == ws[j]) ns = unigramTable(globalData.rg);
				ll += inplaceUpdate(x, ns, lr1 * (1 - zeta), true, coef);
				assert(isnormal(ll));
			}
			llCnt++;
			llSum += ll * (1 - zeta);
		}

		procWords += 1;

		if (zeta > 0 && !fixedWords.count(x))
		{
			llSum += inplaceTimeUpdate(x, lr1 * zeta, coef);
		}

		totalLLCnt += llCnt;
		totalLL += (llSum - llCnt * totalLL) / totalLLCnt;

		if (report && procWords % report == 0)
		{
			float time_per_kword = (procWords - lastProcWords) / timer.getElapsed() / 1000.f;
			printf("%.2f%% %.4f %.4f %.2f kwords/sec\n",
				procWords / (totalWords / 100.f), totalLL, 
				lr1, time_per_kword);
			lastProcWords = procWords;
			timer.reset();
		}
	}
}

void ChronoGramModel::trainVectorsMulti(const uint32_t * ws, size_t N, float timePoint,
	size_t window_length, float start_lr, size_t nEpoch, size_t report, ThreadLocalData& ld)
{
	uniform_int_distribution<size_t> uid{ 0, window_length > 2 ? window_length - 2 : 0 };
	VectorXf coef = makeCoef(L, normalizedTimePoint(timePoint));

	float lr1 = max(start_lr * (1 - procWords / (totalWords + 1.f)), start_lr * 1e-4f);
	float llSum = 0;
	size_t llCnt = 0;

	ld.updateOutIdx.clear();
	ld.updateOutMat = MatrixXf::Zero(M, negativeSampleSize * (window_length + 1) * 2);
	MatrixXf updateIn = MatrixXf::Zero(M, 1);
	MatrixXf updateInBlock = MatrixXf::Zero(M, L);
	for (size_t i = 0; i < N; ++i)
	{
		const auto& x = ws[i];

		size_t jBegin = 0, jEnd = N;
		if (i > window_length) jBegin = i - window_length;
		if (i + window_length < N) jEnd = i + window_length;
		updateIn.setZero();

		// update in, out vector
		for (auto j = jBegin; j < jEnd; ++j)
		{
			if (i == j) continue;
			if (ld.updateOutIdx.find(ws[j]) == ld.updateOutIdx.end())
				ld.updateOutIdx.emplace(ws[j], ld.updateOutIdx.size());

			float ll = getUpdateGradient(x, ws[j], lr1 * (1 - zeta), false, coef,
				updateIn.col(0),
				ld.updateOutMat.col(ld.updateOutIdx[ws[j]]));
			assert(isnormal(ll));
			for (size_t k = 0; k < negativeSampleSize; ++k)
			{
				uint32_t ns = unigramTable(ld.rg);
				while (ns == ws[j]) ns = unigramTable(ld.rg);
				if (ld.updateOutIdx.find(ns) == ld.updateOutIdx.end())
					ld.updateOutIdx.emplace(ns, ld.updateOutIdx.size());

				ll += getUpdateGradient(x, ns, lr1 * (1 - zeta), true, coef,
					updateIn.col(0),
					ld.updateOutMat.col(ld.updateOutIdx[ns]));
				assert(isnormal(ll));
			}
			llCnt++;
			llSum += ll * (1 - zeta);
		}
		
		updateInBlock.setZero();
		if (zeta > 0 && !fixedWords.count(x))
		{
			llSum += getTimeUpdateGradient(x, lr1 * zeta, coef, updateInBlock.block(0, 0, M, L)) * zeta;
		}

		{
			lock_guard<mutex> lock(mtx);
			// deferred update
			if (fixedWords.count(x))
			{
				in.col(x * L) += updateIn.col(0);
			}
			else
			{
				in.block(0, x * L, M, L) += updateIn * VectorXf{ coef.array() * vEta.array() }.transpose();
			}

			if (zeta > 0 && !fixedWords.count(x))
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
	totalLLCnt += llCnt;
	totalLL += (llSum - llCnt * totalLL) / totalLLCnt;
	procWords += N;
	if (report && (procWords - N) / report < procWords / report)
	{
		float time_per_kword = (procWords - lastProcWords) / timer.getElapsed() / 1000.f;
		fprintf(stderr, "%.2f%% %.4f %.4f %.2f kwords/sec\n",
			procWords / (totalWords / 100.f), totalLL,
			lr1, time_per_kword);
		lastProcWords = procWords;
		timer.reset();
	}
}

void ChronoGramModel::trainTimePrior(const float * ts, size_t N, float lr, size_t report)
{
	unordered_map<float, VectorXf> coefMap;
	for (size_t i = 0; i < N; ++i)
	{
		float c_lr = max(lr * (1 - procTimePoints / (totalTimePoints + 1.f)), lr * 1e-4f);
		auto it = coefMap.find(ts[i]);
		if (it == coefMap.end()) it = coefMap.emplace(ts[i], makeCoef(L, normalizedTimePoint(ts[i]))).first;
		float ll = updateTimePrior(c_lr, it->second);
		procTimePoints++;
		timeLLCnt++;
		timeLL += (ll - timeLL) / timeLLCnt;
		if (report && procTimePoints % report == 0)
		{
			fprintf(stderr, "timePrior LL: %.4f\n", timeLL);
		}
	}
}

void ChronoGramModel::normalizeWordDist()
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
	timePriorScale = p;

	for (size_t v = 0; v < vocabs.size(); ++v)
	{
		float p = 0;
		for (size_t i = 0; i <= step; ++i)
		{
			p += 1 - exp(-makeTimedVector(v, coefs[i]).squaredNorm() / 2 * lambda);
		}
		p /= step + 1;
		wordScale[v] = p;
	}
}

float ChronoGramModel::getTimePrior(const Eigen::VectorXf & coef) const
{
	return (1 - exp(-pow(timePrior.dot(coef), 2) / 2)) / timePriorScale;
}

float ChronoGramModel::getWordProbByTime(uint32_t w, const Eigen::VectorXf & timedVector) const
{
	return (1 - exp(-timedVector.squaredNorm() / 2 * lambda)) / wordScale[w];
}

float ChronoGramModel::getWordProbByTime(uint32_t w, float timePoint) const
{
	return getWordProbByTime(w, makeTimedVector(w, makeCoef(L, normalizedTimePoint(timePoint))));
}


void ChronoGramModel::buildVocab(const std::function<ReadResult(size_t)>& reader, size_t minCnt)
{
	float minT = INFINITY, maxT = -INFINITY;
	VocabCounter vc;
	if (thread::hardware_concurrency() > 1)
	{
		bool stop = false;
		mutex inputMtx;
		condition_variable inputCnd;
		queue<vector<string>> workItems;
		thread counter{ [&]()
		{
			while (!stop)
			{
				vector<string> item;
				{
					unique_lock<mutex> l{ inputMtx };
					inputCnd.wait(l, [&]() {return stop || !workItems.empty(); });
					if (stop && workItems.empty()) return;
					item = move(workItems.front());
					workItems.pop();
				}
				vc.update(item.begin(), item.end(),
					VocabCounter::defaultTest, VocabCounter::defaultTrans);
				inputCnd.notify_all();
			}
		} };
		for (size_t id = 0; ; ++id)
		{
			auto res = reader(id);
			if (res.stop) break;
			if (res.words.empty()) continue;
			minT = min(res.timePoint, minT);
			maxT = max(res.timePoint, maxT);
			{
				unique_lock<mutex> l(inputMtx);
				if(workItems.size() >= 4) inputCnd.wait(l, [&]() { return workItems.size() < 4; });
				workItems.emplace(move(res.words));
			}
			inputCnd.notify_all();
		}
		stop = true;
		counter.join();
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
			vc.update(res.words.begin(), res.words.end(), VocabCounter::defaultTest, VocabCounter::defaultTrans);
		}
	}
	zBias = minT;
	zSlope = minT == maxT ? 1 : 1 / (maxT - minT);
	for (size_t i = 0; i < vc.rdict.size(); ++i)
	{
		if (vc.rfreqs[i] < minCnt) continue;
		frequencies.emplace_back(vc.rfreqs[i]);
		vocabs.add(vc.rdict.getStr(i));
	}
	buildModel();
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
	size_t numWorkers, size_t window_length, float start_lr, size_t batch,
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
	timer.reset();
	totalLL = timeLL = 0;
	totalLLCnt = timeLLCnt = 0;

	size_t totW = accumulate(frequencies.begin(), frequencies.end(), 0);
	procWords = lastProcWords = 0;
	// estimate total size
	totalWords = epoch * totW;
	procTimePoints = 0;
	totalTimePoints = totalWords / 4;
	size_t read = 0;
	const auto& procCollection = [&]()
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
				futures.emplace_back(workers.enqueue([&](size_t threadId)
				{
					trainVectorsMulti(d.first.data(), d.first.size(), d.second,
						window_length, start_lr, epoch, report, ld[threadId]);
				}));
			}
			trainTimePrior(timePoints.data(), timePoints.size(), start_lr, report);
			for (auto& f : futures) f.get();
		}
		else
		{
			for (auto& d : collections)
			{
				trainVectors(d.first.data(), d.first.size(), d.second,
					window_length, start_lr, epoch, report);
			}
			trainTimePrior(timePoints.data(), timePoints.size(), start_lr, report);
		}
		collections.clear();
		timePoints.clear();
	};

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
				procCollection();
			}
		}
	}
	procCollection();
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
		len += sqrt((v - u).squaredNorm() + pow(1.f / step, 2));
		v.swap(u);
	}
	return len;
}

vector<tuple<string, float, float>> ChronoGramModel::nearestNeighbors(const string & word,
	float wordTimePoint, float searchingTimePoint, size_t K) const
{
	return mostSimilar({ make_pair(word, wordTimePoint) }, {}, searchingTimePoint, K);
}

vector<tuple<string, float, float>> ChronoGramModel::mostSimilar(
	const vector<pair<string, float>>& positiveWords,
	const vector<pair<string, float>>& negativeWords,
	float searchingTimePoint, size_t K) const
{
	constexpr float threshold = 0.5f;
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
		float wPrior = getWordProbByTime(wv, tv);
		if (wPrior / tPrior < threshold) continue;
		vec += tv;
		uniqs.emplace(wv);
	}

	for (auto& p : negativeWords)
	{
		size_t wv = vocabs.get(p.first);
		if (wv == (size_t)-1) return {};
		VectorXf cf = makeCoef(L, normalizedTimePoint(p.second));
		float tPrior = getTimePrior(cf);
		VectorXf tv = makeTimedVector(wv, cf);
		float wPrior = getWordProbByTime(wv, tv);
		if (wPrior / tPrior < threshold) continue;
		vec -= tv;
		uniqs.emplace(wv);
	}

	vec.normalize();
	float tPrior = getTimePrior(coef);
	vector<tuple<string, float, float>> top;
	vector<pair<float, float>> sim(V);
	for (size_t v = 0; v < V; ++v)
	{
		/*if (uniqs.count(v))
		{
			sim(v) = -INFINITY;
			continue;
		}*/
		VectorXf tv = makeTimedVector(v, coef);
		float wPrior = getWordProbByTime(v, tv);
		if (wPrior / tPrior < threshold)
		{
			sim[v] = make_pair(-INFINITY, wPrior / tPrior);
			continue;
		}
		sim[v] = make_pair(tv.normalized().dot(vec), wPrior / tPrior);
	}

	for (size_t k = 0; k < K; ++k)
	{
		size_t idx = max_element(sim.begin(), sim.end()) - sim.begin();
		top.emplace_back(vocabs.getStr(idx), sim[idx].first, sim[idx].second);
		sim.data()[idx].first = -INFINITY;
	}
	return top;
}

vector<tuple<string, float>> ChronoGramModel::mostSimilar(
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
		if (uniqs.count(v))
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

float ChronoGramModel::similarity(const std::string & word1, float time1, const std::string & word2, float time2) const
{
	size_t wv1 = vocabs.get(word1), wv2 = vocabs.get(word2);
	if (wv1 == (size_t)-1 || wv2 == (size_t)-1) return 0;
	VectorXf c1 = makeCoef(L, normalizedTimePoint(time1)), c2 = makeCoef(L, normalizedTimePoint(time2));
	return makeTimedVector(wv1, c1).normalized().dot(makeTimedVector(wv2, c2).normalized());
}

float ChronoGramModel::similarity(const std::string & word1, const std::string & word2) const
{
	size_t wv1 = vocabs.get(word1), wv2 = vocabs.get(word2);
	if (wv1 == (size_t)-1 || wv2 == (size_t)-1) return 0;
	return out.col(wv1).normalized().dot(out.col(wv2).normalized());
}


ChronoGramModel::LLEvaluater ChronoGramModel::evaluateSent(const std::vector<std::string>& words, size_t windowLen, size_t nsQ) const
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

	return LLEvaluater(*this, windowLen, nsQ, move(wordIds), move(coefs));
}

template<class _XType, class _LLType, class _Func1, class _Func2>
pair<_XType, _LLType> findMaximum(_XType s, _XType e, _XType threshold,
	size_t initStep, size_t maxIteration,
	_LLType gamma, _LLType eta, _LLType mu,
	_Func1 funcLL, _Func2 funcAll)
{
	auto x = s, newx = x;
	_LLType delta = 0;
	_LLType ll = -INFINITY;
	for (size_t c = 0; c < initStep; ++c)
	{
		newx = s + (e - s) * c / initStep;
		_LLType newll = funcLL(newx);
		if (newll > ll)
		{
			ll = newll;
			x = newx;
		}
	}

	for (size_t c = 0; c < maxIteration; ++c)
	{
		delta *= gamma;
		auto p = funcAll(max(min(x + delta, e), s));
		auto newll = get<0>(p);
		/*if (newll < ll)
		{
			cerr << newll << "\t" << ll << endl;
		}*/
		ll = newll;
		auto& dll = get<1>(p);
		auto& ddll = get<2>(p);
		delta += eta * dll / (abs(ddll) + mu);
		newx = max(min(x + delta, e), s);
		if (newx == s || newx == e) delta = 0;
		if (abs(newx - x) < threshold) break;
		x = newx;
	}
	return make_pair(x, ll);
}

pair<float, float> ChronoGramModel::predictSentTime(const std::vector<std::string>& words, size_t windowLen, size_t nsQ, size_t initStep) const
{
	auto evaluator = evaluateSent(words, windowLen, nsQ);
	float maxLL = -INFINITY, maxP = 0;
	auto t = findMaximum(0.f, 1.f, 1e-4f,
		initStep, 0,
		0.2f, 0.8f, 10.f,
		[&](auto x) { return evaluator(x); },
		[&](auto x) { return evaluator.fgh(x); });
	maxLL = t.second, maxP = t.first;
	return make_pair(unnormalizedTimePoint(maxP), maxLL);
}

vector<ChronoGramModel::EvalResult> ChronoGramModel::evaluate(const function<ReadResult(size_t)>& reader,
	const function<void(EvalResult)>& writer,
	size_t numWorkers, size_t windowLen, size_t nsQ, size_t initStep) const
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
			auto p = predictSentTime(words, windowLen, nsQ, initStep);
			lock_guard<mutex> l{ writeMtx };
			res[id] = { time, p.first, p.second, p.second / 2 / windowLen / words.size(), (p.first - time) * zSlope };
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

MatrixXf ChronoGramModel::getEmbedding(const string & word) const
{
	size_t wv = vocabs.get(word);
	if (wv == (size_t)-1) return {};
	return in.block(0, wv * L, M, L);
}

void ChronoGramModel::saveModel(ostream & os) const
{
	writeToBinStream(os, (uint32_t)M);
	writeToBinStream(os, (uint32_t)L);
	writeToBinStream(os, zBias);
	writeToBinStream(os, zSlope);
	vocabs.writeToFile(os);
	writeToBinStream(os, frequencies);
	writeToBinStream(os, in);
	writeToBinStream(os, out);
	writeToBinStream(os, zeta);
	writeToBinStream(os, lambda);
	writeToBinStream(os, timePadding);
	writeToBinStream(os, timePrior);
}

ChronoGramModel ChronoGramModel::loadModel(istream & is)
{
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

float ChronoGramModel::LLEvaluater::operator()(float timePoint) const
{
	const size_t N = wordIds.size(), V = tgm.unigramDist.size();
	auto tCoef = makeCoef(tgm.L, timePoint);
	float ll = 0;
	unordered_map<uint32_t, uint32_t> count;

	for (size_t i = 0; i < N; ++i)
	{
		const uint32_t x = wordIds[i];
		if (x == (uint32_t)-1) continue;
		auto& cx = coefs.find(x)->second;
		size_t jBegin = 0, jEnd = N;
		if (i > windowLen) jBegin = i - windowLen;
		if (i + windowLen < N) jEnd = i + windowLen;

		for (size_t j = jBegin; j < jEnd; ++j)
		{
			if (i == j) continue;
			const uint32_t y = wordIds[j];
			if (y == (uint32_t)-1) continue;
			float d = inner_product(tCoef.data(), tCoef.data() + tgm.L, cx.get(y, nsQ, tgm.L), 0.f);
			ll += logsigmoid(d) * (1 - tgm.zeta);
			count[x]++;
		}

		ll += log(1 - exp(-tgm.makeTimedVector(x, tCoef).squaredNorm() / 2 * tgm.lambda)) * tgm.zeta;
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

tuple<float, float, float> ChronoGramModel::LLEvaluater::fgh(float timePoint) const
{
	const size_t N = wordIds.size(), V = tgm.unigramDist.size();
	constexpr float eps = 1 / 1024.f;
	auto tCoef = makeCoef(tgm.L, timePoint), tDCoef = makeDCoef(tgm.L, timePoint), tDDCoef = makeDCoef(tgm.L, timePoint + eps);
	tDDCoef = ((tDDCoef - tDCoef) / eps).segment(1, tgm.L - 2).eval();

	float ll = 0, dll = 0, ddll = 0;
	unordered_map<uint32_t, uint32_t> count;

	for (size_t i = 0; i < N; ++i)
	{
		const uint32_t x = wordIds[i];
		if (x == (uint32_t)-1) continue;
		auto& cx = coefs.find(x)->second;
		size_t jBegin = 0, jEnd = N;
		if (i > windowLen) jBegin = i - windowLen;
		if (i + windowLen < N) jEnd = i + windowLen;

		for (size_t j = jBegin; j < jEnd; ++j)
		{
			if (i == j) continue;
			const uint32_t y = wordIds[j];
			if (y == (uint32_t)-1) continue;
			float d = inner_product(tCoef.data(), tCoef.data() + tgm.L, cx.get(y, nsQ, tgm.L), 0.f);
			ll += logsigmoid(d) * (1 - tgm.zeta);
			float dd = inner_product(tDCoef.data(), tDCoef.data() + tgm.L - 1, cx.get(y, nsQ, tgm.L) + 1, 0.f), sd;
			dll += dd * (sd = sigmoid(-d)) * (1 - tgm.zeta);
			float ddd = inner_product(tDDCoef.data(), tDDCoef.data() + tgm.L - 2, cx.get(y, nsQ, tgm.L) + 2, 0.f);
			ddll += (ddd * sd - pow(dd, 2) * sd * (1 - sd)) * (1 - tgm.zeta);
			count[x]++;
		}

		auto v = tgm.makeTimedVector(x, tCoef);
		VectorXf dv = tgm.in.block(0, x * tgm.L + 1, tgm.M, tgm.L - 1) * tDCoef;
		float sqn = v.squaredNorm() / 2 * tgm.lambda, sd;
		ll += log(1 - exp(-sqn)) * tgm.zeta; 
		dll += v.dot(dv) * tgm.lambda / (sd = exp(sqn) - 1 + 1e-3f) * tgm.zeta;
		ddll += (tgm.lambda * (v.dot(tgm.in.block(0, x * tgm.L + 2, tgm.M, tgm.L - 2) * tDDCoef) - tgm.lambda * pow(v.dot(dv), 2) + dv.squaredNorm()) / sd
			- pow(tgm.lambda * v.dot(dv) / sd, 2)) * tgm.zeta;
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
				float ddd = inner_product(tDDCoef.data(), tDDCoef.data() + tgm.L - 2, cx.get(j, nsQ, tgm.L) + 2, 0.f);
				ddnll += -tgm.unigramDist[j] * (ddd * sd + pow(dd, 2) * sd * (1 - sd));
				denom += tgm.unigramDist[j];
			}
			ll += nll / denom * tgm.negativeSampleSize * p.second * (1 - tgm.zeta);
			dll += dnll / denom * tgm.negativeSampleSize * p.second * (1 - tgm.zeta);
			ddll += ddnll / denom * tgm.negativeSampleSize * p.second * (1 - tgm.zeta);
		}
	}
	return make_tuple(ll, dll, ddll);
}
