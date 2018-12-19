#include "TimeGramModel.h"
#include "mathUtils.h"
#include "IOUtils.h"
#include "ThreadPool.h"
#include "shiftedLegendre.hpp"
#include <numeric>
#include <iostream>
#include <iterator>
#include <unordered_set>

using namespace std;
using namespace Eigen;

void TimeGramModel::buildModel()
{
	const size_t V = vocabs.size();
	// allocate & initialize model
	in = MatrixXf::Random(M, L * V) * (.5f / M);
	out = MatrixXf::Random(M, V) * (.5f / M);
	
	// build Unigram Table for Negative Sampling
	vector<double> weights;
	transform(frequencies.begin(), frequencies.end(), back_inserter(weights), [](auto w) { return pow(w, 0.75); });
	unigramTable = discrete_distribution<uint32_t>(weights.begin(), weights.end());
}

vector<float> TimeGramModel::makeLegendreCoef(size_t L, float z)
{
	vector<float> coef(L);
	for (size_t i = 0; i < L; ++i)
	{
		coef[i] = slp::slpGet(i, z);
	}
	return coef;
}

VectorXf TimeGramModel::makeTimedVector(size_t wv, const vector<float>& legendreCoef) const
{
	VectorXf vec = VectorXf::Zero(M);
	for (size_t l = 0; l < L; ++l)
	{
		vec += legendreCoef[l] * in.col(wv * L + l);
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
	size_t window_length, float start_lr)
{
	uniform_int_distribution<size_t> uid{ 0, window_length > 2 ? window_length - 2 : 0 };
	vector<float> legendreCoef = makeLegendreCoef(L, (timePoint - zBias) / zSlope);

	for (size_t i = 0; i < N; ++i)
	{
		const auto& x = ws[i];
		float lr1 = max(start_lr * (1 - procWords / (totalWords + 1.f)), start_lr * 1e-4f);
		float lr2 = lr1;

		int random_reduce = context_cut ? uid(globalData.rg) : 0;
		int window = window_length - random_reduce;
		size_t jBegin = 0, jEnd = N;
		if (i > window) jBegin = i - window;
		if (i + window < N) jEnd = i + window;

		// update in, out vector
		for (auto j = jBegin; j < jEnd; ++j)
		{
			if (i == j) continue;
			float ll = inplaceUpdate(x, ws[j], lr1, false, legendreCoef);
			assert(isnormal(ll));
			for (size_t k = 0; k < negativeSampleSize; ++k)
			{
				uint32_t ns = unigramTable(globalData.rg);
				while (ns == ws[j]) ns = unigramTable(globalData.rg);
				ll += inplaceUpdate(x, ns, lr1, true, legendreCoef);
				assert(isnormal(ll));
			}
			totalLLCnt++;
			totalLL += (ll - totalLL) / totalLLCnt;
		}

		procWords += 1;

		if (procWords % 10000 == 0)
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
	size_t window_length, float start_lr, ThreadLocalData& ld)
{
	uniform_int_distribution<size_t> uid{ 0, window_length > 2 ? window_length - 2 : 0 };
	vector<float> legendreCoef = makeLegendreCoef(L, (timePoint - zBias) / zSlope);
	float llSum = 0;
	size_t llCnt = 0;

	ld.updateOutIdx.clear();
	ld.updateOutMat = MatrixXf::Zero(M, negativeSampleSize * (window_length + 1) * 2);
	for (size_t i = 0; i < N; ++i)
	{
		const auto& x = ws[i];
		float lr1 = max(start_lr * (1 - procWords / (totalWords + 1.f)), start_lr * 1e-4f);
		float lr2 = lr1;

		int random_reduce = context_cut ? uid(ld.rg) : 0;
		int window = window_length - random_reduce;
		size_t jBegin = 0, jEnd = N;
		if (i > window) jBegin = i - window;
		if (i + window < N) jEnd = i + window;
		MatrixXf updateIn = MatrixXf::Zero(M, 1);

		// update in, out vector
		for (auto j = jBegin; j < jEnd; ++j)
		{
			if (i == j) continue;
			if (ld.updateOutIdx.find(ws[j]) == ld.updateOutIdx.end()) 
				ld.updateOutIdx.emplace(ws[j], ld.updateOutIdx.size());

			float ll = getUpdateGradient(x, ws[j], lr1, false, legendreCoef,
				updateIn.col(0),
				ld.updateOutMat.col(ld.updateOutIdx[ws[j]]));
			assert(isnormal(ll));
			for (size_t k = 0; k < negativeSampleSize; ++k)
			{
				uint32_t ns = unigramTable(ld.rg);
				while (ns == ws[j]) ns = unigramTable(ld.rg);
				if (ld.updateOutIdx.find(ns) == ld.updateOutIdx.end()) 
					ld.updateOutIdx.emplace(ns, ld.updateOutIdx.size());

				ll += getUpdateGradient(x, ns, lr1, true, legendreCoef,
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
				in.col(x * L + l) += legendreCoef[l] * updateIn.col(0);
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
	if ((procWords - N) / 10000 < procWords / 10000)
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
	size_t numWorkers, size_t window_length, float start_lr, size_t batch, size_t epoch)
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
					trainVectorsMulti(d.first.data(), d.first.size(), d.second, window_length, start_lr, ld[threadId]);
				}));
			}
			for (auto& f : futures) f.get();
		}
		else
		{
			for (auto& d : collections)
			{
				trainVectors(d.first.data(), d.first.size(), d.second, window_length, start_lr);
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

vector<tuple<string, float>> TimeGramModel::nearestNeighbors(const string & word, 
	float timePoint, size_t K) const
{
	const size_t V = vocabs.size();
	size_t wv = vocabs.get(word);
	if (wv == (size_t)-1) return {};
	vector<float> legendreCoef = makeLegendreCoef(L, (timePoint - zBias) / zSlope);
	VectorXf vec = makeTimedVector(wv, legendreCoef).normalized();

	vector<tuple<string, float>> top;
	VectorXf sim(V);
	for (size_t v = 0; v < V; ++v)
	{
		if (v == wv)
		{
			sim(v) = -INFINITY;
			continue;
		}
		sim(v) = makeTimedVector(v, legendreCoef).normalized().dot(vec);
	}

	for (size_t k = 0; k < K; ++k)
	{
		size_t idx = max_element(sim.data(), sim.data() + sim.size()) - sim.data();
		top.emplace_back(vocabs.getStr(idx), sim.data()[idx]);
		sim.data()[idx] = -INFINITY;
	}
	return top;
}

vector<tuple<string, float>> TimeGramModel::mostSimilar(const vector<string>& positiveWords, 
	const vector<string>& negativeWords, float timePoint, size_t K) const
{
	VectorXf vec = VectorXf::Zero(M);
	const size_t V = vocabs.size();
	unordered_set<size_t> uniqs;
	vector<float> legendreCoef = makeLegendreCoef(L, (timePoint - zBias) / zSlope);
	for (auto& p : positiveWords)
	{
		size_t wv = vocabs.get(p);
		if (wv == (size_t)-1) return {};
		vec += makeTimedVector(wv, legendreCoef);
		uniqs.emplace(wv);
	}

	for (auto& p : negativeWords)
	{
		size_t wv = vocabs.get(p);
		if (wv == (size_t)-1) return {};
		vec -= makeTimedVector(wv, legendreCoef);
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
		sim(v) = makeTimedVector(v, legendreCoef).normalized().dot(vec);
	}

	for (size_t k = 0; k < K; ++k)
	{
		size_t idx = max_element(sim.data(), sim.data() + sim.size()) - sim.data();
		top.emplace_back(vocabs.getStr(idx), sim.data()[idx]);
		sim.data()[idx] = -INFINITY;
	}
	return top;
}

void TimeGramModel::saveModel(ostream & os) const
{
	writeToBinStream(os, (uint32_t)M);
	writeToBinStream(os, (uint32_t)L);
	writeToBinStream(os, (uint32_t)context_cut);
	vocabs.writeToFile(os);
	writeToBinStream(os, frequencies);
	writeToBinStream(os, in);
	writeToBinStream(os, out);
}

TimeGramModel TimeGramModel::loadModel(istream & is)
{
	size_t M = readFromBinStream<uint32_t>(is);
	size_t L = readFromBinStream<uint32_t>(is);
	bool context_cut = readFromBinStream<uint32_t>(is);
	TimeGramModel ret{ M, L };
	ret.context_cut = context_cut;
	ret.vocabs.readFromFile(is);
	size_t V = ret.vocabs.size();
	ret.in.resize(M, L * V);
	ret.out.resize(M, V);

	readFromBinStream(is, ret.frequencies);
	readFromBinStream(is, ret.in);
	readFromBinStream(is, ret.out);

	return ret;
}