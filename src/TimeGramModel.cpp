#include <numeric>
#include <iostream>
#include <iterator>
#include <unordered_set>
#include "TimeGramModel.h"
#include "mathUtils.h"
#include "IOUtils.h"
#include "ThreadPool.h"
#include "polynomials.hpp"

using namespace std;
using namespace Eigen;

void TimeGramModel::buildModel()
{
	const size_t V = vocabs.size();
	// allocate & initialize model
	in = MatrixXf::Random(M, L * V) * (.5f / M);
	out = MatrixXf::Random(M, V) * (.5f / M);
	
	buildTable();
}

void TimeGramModel::buildTable()
{
	// build Unigram Table for Negative Sampling
	vector<double> weights;
	transform(frequencies.begin(), frequencies.end(), back_inserter(weights), [](auto w) { return pow(w, 0.75); });
	unigramTable = discrete_distribution<uint32_t>(weights.begin(), weights.end());
	auto p = unigramTable.probabilities();
	unigramDist = vector<float>{ p.begin(), p.end() };
}

vector<float> TimeGramModel::makeCoef(size_t L, float z)
{
	vector<float> coef(L);
	for (size_t i = 0; i < L; ++i)
	{
		coef[i] = poly::chebyshevTGet(i, 2 * z - 1);
	}
	return coef;
}

std::vector<float> TimeGramModel::makeDCoef(size_t L, float z)
{
	vector<float> coef(L - 1);
	for (size_t i = 1; i < L; ++i)
	{
		coef[i - 1] = 2 * poly::chebyshevTDerived(i, 2 * z - 1);
	}
	return coef;
}

VectorXf TimeGramModel::makeTimedVector(size_t wv, const vector<float>& coef) const
{
	VectorXf vec = VectorXf::Zero(M);
	for (size_t l = 0; l < L; ++l)
	{
		vec += coef[l] * in.col(wv * L + l);
	}
	return vec;
}

float TimeGramModel::inplaceUpdate(size_t x, size_t y, float lr, bool negative, const std::vector<float>& lWeight)
{
	auto outcol = out.col(y);
	VectorXf inSum = VectorXf::Zero(M);
	for (size_t l = 0; l < L; ++l)
	{
		inSum += lWeight[l] * in.col(x * L + l);
	}
	float f = inSum.dot(outcol);
	float pr = logsigmoid(f * (negative ? -1 : 1));

	float d = (negative ? 0 : 1) - sigmoid(f);
	float g = lr * d;

	VectorXf in_grad = g * outcol;
	VectorXf out_grad = g * inSum;
	outcol += out_grad;

	for (size_t l = 0; l < L; ++l)
	{
		in.col(x * L + l) += lWeight[l] * in_grad;
	}
	return pr;
}

float TimeGramModel::getUpdateGradient(size_t x, size_t y, float lr, bool negative, const std::vector<float>& lWeight,
	Eigen::DenseBase<Eigen::MatrixXf>::ColXpr xGrad, Eigen::DenseBase<Eigen::MatrixXf>::ColXpr yGrad)
{
	auto outcol = out.col(y);
	VectorXf inSum = VectorXf::Zero(M);
	for (size_t l = 0; l < L; ++l)
	{
		inSum += lWeight[l] * in.col(x * L + l);
	}
	float f = inSum.dot(outcol);
	float pr = logsigmoid(f * (negative ? -1 : 1));

	float d = (negative ? 0 : 1) - sigmoid(f);
	float g = lr * d;

	xGrad += g * outcol;
	yGrad += g * inSum;
	return max(pr, -100.f);
}



void TimeGramModel::trainVectors(const uint32_t * ws, size_t N, float timePoint,
	size_t window_length, float start_lr, size_t report)
{
	uniform_int_distribution<size_t> uid{ 0, window_length > 2 ? window_length - 2 : 0 };
	vector<float> coef = makeCoef(L, normalizeTimePoint(timePoint));

	for (size_t i = 0; i < N; ++i)
	{
		const auto& x = ws[i];
		float lr1 = max(start_lr * (1 - procWords / (totalWords + 1.f)), start_lr * 1e-4f);
		float lr2 = lr1;

		size_t jBegin = 0, jEnd = N;
		if (i > window_length) jBegin = i - window_length;
		if (i + window_length < N) jEnd = i + window_length;

		// update in, out vector
		for (auto j = jBegin; j < jEnd; ++j)
		{
			if (i == j) continue;
			float ll = inplaceUpdate(x, ws[j], lr1, false, coef);
			assert(isnormal(ll));
			for (size_t k = 0; k < negativeSampleSize; ++k)
			{
				uint32_t ns = unigramTable(globalData.rg);
				while (ns == ws[j]) ns = unigramTable(globalData.rg);
				ll += inplaceUpdate(x, ns, lr1, true, coef);
				assert(isnormal(ll));
			}
			totalLLCnt++;
			totalLL += (ll - totalLL) / totalLLCnt;
		}

		procWords += 1;

		if (report && procWords % report == 0)
		{
			float time_per_kword = (procWords - lastProcWords) / timer.getElapsed() / 1000.f;
			printf("%.2f%% %.4f %.4f %.4f %.2f kwords/sec\n",
				procWords / (totalWords / 100.f), totalLL, lr1, lr2,
				time_per_kword);
			lastProcWords = procWords;
			timer.reset();
		}
	}
}


void TimeGramModel::trainVectorsMulti(const uint32_t * ws, size_t N, float timePoint,
	size_t window_length, float start_lr, size_t report, ThreadLocalData& ld)
{
	uniform_int_distribution<size_t> uid{ 0, window_length > 2 ? window_length - 2 : 0 };
	vector<float> coef = makeCoef(L, normalizeTimePoint(timePoint));
	float llSum = 0;
	size_t llCnt = 0;

	ld.updateOutIdx.clear();
	ld.updateOutMat = MatrixXf::Zero(M, negativeSampleSize * (window_length + 1) * 2);
	for (size_t i = 0; i < N; ++i)
	{
		const auto& x = ws[i];
		float lr1 = max(start_lr * (1 - procWords / (totalWords + 1.f)), start_lr * 1e-4f);
		float lr2 = lr1;

		size_t jBegin = 0, jEnd = N;
		if (i > window_length) jBegin = i - window_length;
		if (i + window_length < N) jEnd = i + window_length;
		MatrixXf updateIn = MatrixXf::Zero(M, 1);

		// update in, out vector
		for (auto j = jBegin; j < jEnd; ++j)
		{
			if (i == j) continue;
			if (ld.updateOutIdx.find(ws[j]) == ld.updateOutIdx.end()) 
				ld.updateOutIdx.emplace(ws[j], ld.updateOutIdx.size());

			float ll = getUpdateGradient(x, ws[j], lr1, false, coef,
				updateIn.col(0),
				ld.updateOutMat.col(ld.updateOutIdx[ws[j]]));
			assert(isnormal(ll));
			for (size_t k = 0; k < negativeSampleSize; ++k)
			{
				uint32_t ns = unigramTable(ld.rg);
				while (ns == ws[j]) ns = unigramTable(ld.rg);
				if (ld.updateOutIdx.find(ns) == ld.updateOutIdx.end()) 
					ld.updateOutIdx.emplace(ns, ld.updateOutIdx.size());

				ll += getUpdateGradient(x, ns, lr1, true, coef,
					updateIn.col(0),
					ld.updateOutMat.col(ld.updateOutIdx[ns]));
				assert(isnormal(ll));
			}
			llCnt++;
			llSum += ll;
		}

		{
			lock_guard<mutex> lock(mtx);
			for (size_t l = 0; l < L; ++l)
			{
				in.col(x * L + l) += coef[l] * updateIn.col(0);
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
			max(start_lr * (1 - procWords / (totalWords + 1.f)), start_lr * 1e-4f),
			time_per_kword);
		lastProcWords = procWords;
		timer.reset();
	}
}
void TimeGramModel::buildVocab(const std::function<ReadResult(size_t)>& reader, size_t minCnt)
{
	VocabCounter vc;
	float minT = INFINITY, maxT = -INFINITY;
	for(size_t id = 0; ; ++id)
	{
		auto res = reader(id);
		if (res.stop) break;
		if (res.words.empty()) continue;
		minT = min(res.timePoint, minT);
		maxT = max(res.timePoint, maxT);
		vc.update(res.words.begin(), res.words.end(), VocabCounter::defaultTest, VocabCounter::defaultTrans);
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

void TimeGramModel::train(const function<ReadResult(size_t)>& reader,
	size_t numWorkers, size_t window_length, float start_lr, size_t batch, size_t epoch, size_t report)
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
	timer.reset();
	totalLL = 0;
	totalLLCnt = 0;
	size_t totW = accumulate(frequencies.begin(), frequencies.end(), 0);
	procWords = lastProcWords = 0;
	// estimate total size
	totalWords = epoch * totW;
	size_t read = 0;
	const auto& procCollection = [&]()
	{
		if (collections.empty()) return;
		shuffle(collections.begin(), collections.end(), globalData.rg);
		if (numWorkers > 1)
		{
			vector<future<void>> futures;
			futures.reserve(collections.size());
			for (auto& d : collections)
			{
				futures.emplace_back(workers.enqueue([&](size_t threadId)
				{
					trainVectorsMulti(d.first.data(), d.first.size(), d.second, window_length, start_lr, report, ld[threadId]);
				}));
			}
			for (auto& f : futures) f.get();
		}
		else
		{
			for (auto& d : collections)
			{
				trainVectors(d.first.data(), d.first.size(), d.second, window_length, start_lr, report);
			}
		}
		collections.clear();
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
					procWords += 1;
					continue;
				}
				doc.emplace_back(id);
			}

			if (doc.size() < 3)
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
}

float TimeGramModel::arcLengthOfWord(const string & word, size_t step) const
{
	size_t wv = vocabs.get(word);
	if (wv == (size_t)-1) return {};
	float len = 0;
	VectorXf v = makeTimedVector(wv, makeCoef(L, 0));
	for (size_t i = 0; i < step; ++i)
	{
		VectorXf u = makeTimedVector(wv, makeCoef(L, (float)(i+1) / step));
		len += sqrt((v - u).squaredNorm() + pow(1.f / step, 2));
		v.swap(u);
	}
	return len;
}

vector<tuple<string, float>> TimeGramModel::nearestNeighbors(const string & word,
	float wordTimePoint, float searchingTimePoint, size_t K) const
{
	const size_t V = vocabs.size();
	size_t wv = vocabs.get(word);
	if (wv == (size_t)-1) return {};
	vector<float> coef = makeCoef(L, normalizeTimePoint(searchingTimePoint));
	VectorXf vec = makeTimedVector(wv, makeCoef(L, normalizeTimePoint(wordTimePoint))).normalized();

	vector<tuple<string, float>> top;
	VectorXf sim(V);
	for (size_t v = 0; v < V; ++v)
	{
		if (v == wv)
		{
			sim(v) = -INFINITY;
			continue;
		}
		sim(v) = makeTimedVector(v, coef).normalized().dot(vec);
	}

	for (size_t k = 0; k < K; ++k)
	{
		size_t idx = max_element(sim.data(), sim.data() + sim.size()) - sim.data();
		top.emplace_back(vocabs.getStr(idx), sim.data()[idx]);
		sim.data()[idx] = -INFINITY;
	}
	return top;
}

vector<tuple<string, float>> TimeGramModel::mostSimilar(
	const vector<pair<string, float>>& positiveWords, 
	const vector<pair<string, float>>& negativeWords, 
	float searchingTimePoint, size_t K) const
{
	VectorXf vec = VectorXf::Zero(M);
	const size_t V = vocabs.size();
	unordered_set<size_t> uniqs;
	vector<float> coef = makeCoef(L, normalizeTimePoint(searchingTimePoint));
	for (auto& p : positiveWords)
	{
		size_t wv = vocabs.get(p.first);
		if (wv == (size_t)-1) return {};
		vec += makeTimedVector(wv, makeCoef(L, normalizeTimePoint(p.second)));
		uniqs.emplace(wv);
	}

	for (auto& p : negativeWords)
	{
		size_t wv = vocabs.get(p.first);
		if (wv == (size_t)-1) return {};
		vec -= makeTimedVector(wv, makeCoef(L, normalizeTimePoint(p.second)));
		uniqs.emplace(wv);
	}

	vec.normalize();

	vector<tuple<string, float>> top;
	VectorXf sim(V);
	for (size_t v = 0; v < V; ++v)
	{
		/*if (uniqs.count(v))
		{
			sim(v) = -INFINITY;
			continue;
		}*/
		sim(v) = makeTimedVector(v, coef).normalized().dot(vec);
	}

	for (size_t k = 0; k < K; ++k)
	{
		size_t idx = max_element(sim.data(), sim.data() + sim.size()) - sim.data();
		top.emplace_back(vocabs.getStr(idx), sim.data()[idx]);
		sim.data()[idx] = -INFINITY;
	}
	return top;
}

TimeGramModel::LLEvaluater TimeGramModel::evaluateSent(const std::vector<std::string>& words, size_t windowLen, size_t nsQ) const
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
		if(nsQ) coefs[i].dataVec.resize((V + nsQ - 1) / nsQ * L);
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

	return LLEvaluater(L, negativeSampleSize, windowLen, nsQ, move(wordIds), move(coefs), unigramDist);
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
		ll = get<0>(p);
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

pair<float, float> TimeGramModel::predictSentTime(const std::vector<std::string>& words, size_t windowLen, size_t nsQ, size_t initStep) const
{
	auto evaluator = evaluateSent(words, windowLen, nsQ);
	float maxLL = -INFINITY, maxP = 0;
	auto t = findMaximum(0.f, 1.f, 1e-4f,
		initStep, 10,
		0.2f, 0.8f, 10.f, 
		[&](auto x) { return evaluator(x); },
		[&](auto x) { return evaluator.fgh(x); });
	maxLL = t.second, maxP = t.first;
	/*constexpr size_t SLICE = 32;
	for (size_t i = 0; i <= SLICE; ++i)
	{
		auto ll = evaluator(i / (float)SLICE);
		if (ll > maxLL)
		{
			if(fmod(maxP * SLICE, 1)) cout << maxP << '\t' << maxLL << " --> " << i / (float)SLICE << '\t' << ll << endl;
			maxLL = ll;
			maxP = i / (float)SLICE;
		}
	}*/
	return make_pair(unnormalizeTimePoint(maxP), maxLL);
}

void TimeGramModel::saveModel(ostream & os) const
{
	writeToBinStream(os, (uint32_t)M);
	writeToBinStream(os, (uint32_t)L);
	writeToBinStream(os, zBias);
	writeToBinStream(os, zSlope);
	vocabs.writeToFile(os);
	writeToBinStream(os, frequencies);
	writeToBinStream(os, in);
	writeToBinStream(os, out);
}

TimeGramModel TimeGramModel::loadModel(istream & is)
{
	size_t M = readFromBinStream<uint32_t>(is);
	size_t L = readFromBinStream<uint32_t>(is);
	TimeGramModel ret{ M, L };
	ret.zBias = readFromBinStream<float>(is);
	ret.zSlope = readFromBinStream<float>(is);
	ret.vocabs.readFromFile(is);
	size_t V = ret.vocabs.size();
	ret.in.resize(M, L * V);
	ret.out.resize(M, V);

	readFromBinStream(is, ret.frequencies);
	readFromBinStream(is, ret.in);
	readFromBinStream(is, ret.out);
	ret.buildTable();
	return ret;
}

float TimeGramModel::LLEvaluater::operator()(float timePoint) const
{
	const size_t N = wordIds.size(), V = unigramDist.size();
	auto tCoef = makeCoef(L, timePoint);
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
			float d = inner_product(tCoef.begin(), tCoef.end(), cx.get(y, nsQ, L), 0.f);
			ll += logsigmoid(d);
			count[x]++;
		}
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
				float d = inner_product(tCoef.begin(), tCoef.end(), cx.get(j, nsQ, L), 0.f);
				nll += unigramDist[j] * logsigmoid(-d);
				denom += unigramDist[j];
			}
			ll += nll / denom * negativeSampleSize * p.second;
		}
	}
	return ll;
}

tuple<float, float, float> TimeGramModel::LLEvaluater::fgh(float timePoint) const
{
	const size_t N = wordIds.size(), V = unigramDist.size();
	constexpr float eps = 1 / 1024.f;
	auto tCoef = makeCoef(L, timePoint), tDCoef = makeDCoef(L, timePoint), tDDCoef = makeDCoef(L, timePoint + eps);
	for (size_t i = 1; i < tDDCoef.size(); ++i)
	{
		tDDCoef[i - 1] = (tDDCoef[i] - tDCoef[i]) / eps;
	}
	tDDCoef.pop_back();

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
			float d = inner_product(tCoef.begin(), tCoef.end(), cx.get(y, nsQ, L), 0.f);
			ll += logsigmoid(d);
			float dd = inner_product(tDCoef.begin(), tDCoef.end(), cx.get(y, nsQ, L) + 1, 0.f), sd;
			dll += dd * (sd = sigmoid(-d));
			float ddd = inner_product(tDDCoef.begin(), tDDCoef.end(), cx.get(y, nsQ, L) + 2, 0.f);
			ddll += ddd * sd - pow(dd, 2) * sd * (1 - sd);
			count[x]++;
		}
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
				float d = inner_product(tCoef.begin(), tCoef.end(), cx.get(j, nsQ, L), 0.f);
				nll += unigramDist[j] * logsigmoid(-d);
				float dd = inner_product(tDCoef.begin(), tDCoef.end(), cx.get(j, nsQ, L) + 1, 0.f), sd;
				dnll += -unigramDist[j] * dd * (sd = sigmoid(d));
				float ddd = inner_product(tDDCoef.begin(), tDDCoef.end(), cx.get(j, nsQ, L) + 2, 0.f);
				ddnll += -unigramDist[j] * (ddd * sd + pow(dd, 2) * sd * (1 - sd));
				denom += unigramDist[j];
			}
			ll += nll / denom * negativeSampleSize * p.second;
			dll += dnll / denom * negativeSampleSize * p.second;
			ddll += ddnll / denom * negativeSampleSize * p.second;
		}
	}
	return make_tuple(ll, dll, ddll);
}
