#include "TimeGramModel.h"
#include "mathUtils.h"
#include "IOUtils.h"
#include "ThreadPool.h"
#include <numeric>
#include <iostream>
#include <iterator>
#include <unordered_set>

using namespace std;
using namespace Eigen;

struct HierarchicalSoftmaxNode
{
	uint32_t parent = -1;
	bool branch = false;

	static vector<pair<int32_t, int8_t>> softmax_path(
		const vector<HierarchicalSoftmaxNode>& nodes, size_t V, size_t id)
	{
		vector<pair<int32_t, int8_t>> ret;
		while (nodes[id].parent != (uint32_t)-1)
		{
			const auto& node = nodes[id];
			assert(node.parent >= V);
			ret.emplace_back(node.parent - V, node.branch ? 1 : -1);
			id = node.parent;
		}
		return ret;
	}
};


vector<TimeGramModel::HierarchicalOuput> TimeGramModel::buildHuffmanTree() const
{
	auto V = vocabs.size();
	auto nodes = vector<HierarchicalSoftmaxNode>(V);
	vector<pair<size_t, size_t>> heap;

	for (size_t i = 0; i < V; ++i)
	{
		heap.emplace_back(i, frequencies[i]);
	}
	size_t heapSize = V;

	const auto& freq_cmp = [](const pair<size_t, size_t>& a, const pair<size_t, size_t>& b)
	{
		return a.second > b.second;
	};
	make_heap(heap.begin(), heap.end(), freq_cmp);

	size_t L = V;
	while (heapSize > 1)
	{
		nodes.emplace_back();
		size_t freq = 0;

		pop_heap(heap.begin(), heap.begin() + heapSize, freq_cmp);
		heapSize--;
		nodes[heap[heapSize].first].parent = L;
		nodes[heap[heapSize].first].branch = true;
		freq += heap[heapSize].second;

		pop_heap(heap.begin(), heap.begin() + heapSize, freq_cmp);
		heapSize--;
		nodes[heap[heapSize].first].parent = L;
		nodes[heap[heapSize].first].branch = false;
		freq += heap[heapSize].second;

		heap[heapSize] = { nodes.size() - 1, freq };
		heapSize++;
		push_heap(heap.begin(), heap.begin() + heapSize, freq_cmp);
		L++;
	}


	auto outputs = vector<HierarchicalOuput>(V);
	for (auto v = 0; v < V; ++v)
	{
		vector<int8_t> code;
		vector<uint32_t> path;
		for (auto& p : HierarchicalSoftmaxNode::softmax_path(nodes, V, v))
		{
			code.emplace_back(p.second);
			path.emplace_back(p.first);
		}
		outputs[v] = HierarchicalOuput{ code, path };
	}
	return outputs;
}

void TimeGramModel::buildModel()
{
	const size_t V = vocabs.size();
	// allocate & initialize model
	in = MatrixXf::Random(M, T * V) * (.5f / M);
	out = MatrixXf::Random(M, V) * (.5f / M);
	counts = MatrixXf::Zero(T, V);
	for (size_t v = 0; v < V; ++v) counts(0, v) = frequencies[v];

	// build Huffman Tree for Hierarchical Softmax
	if (mode == Mode::hierarchicalSoftmax)
	{
		auto outputs = buildHuffmanTree();
		size_t max_length = max_element(outputs.begin(), outputs.end(), [](const HierarchicalOuput& a, const HierarchicalOuput& b)
		{
			return a.code.size() < b.code.size();
		})->code.size();
		path = Matrix<uint32_t, Dynamic, Dynamic>::Zero(max_length, V);
		code = Matrix<int8_t, Dynamic, Dynamic>::Zero(max_length, V);

		for (size_t v = 0; v < V; ++v)
		{
			copy(outputs[v].code.begin(), outputs[v].code.end(), code.col(v).data());
			copy(outputs[v].path.begin(), outputs[v].path.end(), path.col(v).data());
		}
	}
	// build Unigram Table for Negative Sampling
	else
	{
		vector<double> weights;
		transform(frequencies.begin(), frequencies.end(), back_inserter(weights), [](auto w) { return pow(w, 0.75); });
		unigramTable = discrete_distribution<uint32_t>(weights.begin(), weights.end());
	}
}

pair<VectorXf, size_t> TimeGramModel::getExpectedLogPi(size_t v) const
{
	float vAlpha = alpha; // *pow(log(frequencies[v]), alphaFreqWeight);
	VectorXf pi = counts.col(v);
	size_t senses = 0;
	float r = 0.f, x = 1.f;
	double ts = pi.sum();
	for (size_t k = 0; k < T - 1; ++k)
	{
		ts = max(ts - pi[k], 0.);
		float a = 1 + pi[k] - d, b = vAlpha + (k + 1) * d + ts;
		pi[k] = meanlog_beta(a, b) + r;
		r += meanlog_mirror(a, b);

		float pi_k = mean_beta(a, b) * x;
		x = max(x - pi_k, 0.f);
		if (pi_k >= min_prob) senses++;
	}
	pi[T - 1] = r;
	if (x >= min_prob) senses++;
	return make_pair(move(pi), senses);

}

pair<VectorXf, size_t> TimeGramModel::getExpectedPi(size_t v) const
{
	float vAlpha = alpha; // *pow(log(frequencies[v]), alphaFreqWeight);
	VectorXf pi(T);
	size_t senses = 0;
	float r = 1.f;
	float ts = counts.col(v).sum();
	for (size_t k = 0; k < T - 1; ++k)
	{
		ts = max(ts - counts(k, v), 0.f);
		float a = 1 + counts(k, v) - d, b = vAlpha + (k + 1) * d + ts;
		pi[k] = mean_beta(a, b) * r;
		if (pi[k] >= min_prob) senses++;
		r = max(r - pi[k], 0.f);
	}
	pi[T - 1] = r;
	if (r >= min_prob) senses++;
	return make_pair(move(pi), senses);
}


void TimeGramModel::updateZ(VectorXf & z, size_t x, size_t y) const
{
	for (int n = 0; n < code.rows() && code(n, y); ++n)
	{
		auto ocol = out.col(path(n, y));
		for (int k = 0; k < T; ++k)
		{
			float f = in.col(x * T + k).dot(ocol);
			z[k] += logsigmoid(f * code(n, y));
		}
	}
}

float TimeGramModel::inplaceUpdate(size_t x, size_t y, const VectorXf& z, float lr)
{
	MatrixXf in_grad = MatrixXf::Zero(M, T);
	VectorXf out_grad = VectorXf::Zero(M);

	float pr = 0;
	for (int n = 0; n < code.rows() && code(n, y); ++n)
	{
		auto outcol = out.col(path(n, y));
		out_grad.setZero();

		for (int k = 0; k < T; ++k)
		{
			if (z[k] < sense_threshold) continue;
			auto incol = in.col(x * T + k);
			float f = incol.dot(outcol);
			pr += z[k] * logsigmoid(f * code(n, y));

			float d = (1 + code(n, y)) / 2 - sigmoid(f);
			float g = z[k] * lr * d;

			in_grad.col(k) += g * outcol;
			out_grad += g * incol;
		}
		outcol += out_grad;
	}

	for (int k = 0; k < T; ++k)
	{
		if (z[k] < sense_threshold) continue;
		auto incol = in.col(x * T + k);
		incol += in_grad.col(k);
	}
	return pr;
}

void TimeGramModel::updateZ_NS(VectorXf & z, size_t x, size_t y, bool negative) const
{
	for (int k = 0; k < T; ++k)
	{
		float f = in.col(x * T + k).dot(out.col(y));
		z[k] += logsigmoid(f * (negative ? -1 : 1));
	}
}

float TimeGramModel::inplaceUpdate_NS(size_t x, size_t y, const VectorXf & z, float lr, bool negative)
{
	MatrixXf in_grad = MatrixXf::Zero(M, T);
	VectorXf out_grad = VectorXf::Zero(M);

	float pr = 0;
	auto outcol = out.col(y);
	for (int k = 0; k < T; ++k)
	{
		if (z[k] < sense_threshold) continue;
		auto incol = in.col(x * T + k);
		float f = incol.dot(outcol);
		pr += z[k] * logsigmoid(f * (negative ? -1 : 1));

		float d = (negative ? 0 : 1) - sigmoid(f);
		float g = z[k] * lr * d;

		in_grad.col(k) += g * outcol;
		out_grad += g * incol;
	}
	outcol += out_grad;

	for (int k = 0; k < T; ++k)
	{
		if (z[k] < sense_threshold) continue;
		in.col(x * T + k) += in_grad.col(k);
	}
	return pr;
}

void TimeGramModel::updateCounts(size_t x, const VectorXf & localCounts, float lr)
{
	counts.col(x) += lr * (localCounts * frequencies[x] - counts.col(x).eval());
}

inline void exp_normalize(VectorXf& z)
{
	float max = z.maxCoeff();
	z = (z - VectorXf::Constant(z.size(), max)).array().exp();
	z /= z.sum();
}

void TimeGramModel::trainVectors(const uint32_t * ws, size_t N, size_t window_length, float start_lr,
	ThreadLocalData& ld, size_t threadId)
{
	size_t senses = 0, max_senses = 0;
	uniform_int_distribution<size_t> uid{ 0, window_length > 2 ? window_length - 2 : 0 };
	bernoulli_distribution bd{ .1 };
	vector<uint32_t> negativeSamples;
	negativeSamples.reserve(negativeSampleSize);
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

		// sample negative examples, which is not included in positive
		if (mode != Mode::hierarchicalSoftmax)
		{
			unordered_set<uint32_t> positiveSamples;
			positiveSamples.emplace(x);
			for (auto j = jBegin; j < jEnd; ++j)
			{
				if (i == j) continue;
				positiveSamples.emplace(ws[j]);
			}
			negativeSamples.clear();
			while (negativeSamples.size() < negativeSampleSize)
			{
				auto nw = unigramTable(ld.rg);
				if (positiveSamples.count(nw)) continue;
				negativeSamples.emplace_back(nw);
			}
		}

		lock_guard<mutex> lock(mtx);

		// updating z, which represents probabilities of each sense
		auto t = getExpectedLogPi(x);
		VectorXf& z = t.first;
		senses += t.second;
		max_senses = max(max_senses, t.second);

		// hierarchical softmax
		if (mode == Mode::hierarchicalSoftmax)
		{
			for (auto j = jBegin; j < jEnd; ++j)
			{
				if (i == j) continue;
				updateZ(z, x, ws[j]);
				assert(isnormal(z[0]));
			}
		}
		// negative sampling
		else
		{
			for (auto j = jBegin; j < jEnd; ++j)
			{
				if (i == j) continue;
				updateZ_NS(z, x, ws[j], false);
				assert(isnormal(z[0]));
			}
			for (auto ns : negativeSamples)
			{
				updateZ_NS(z, x, ns, true);
				assert(isnormal(z[0]));
			}
		}

		exp_normalize(z);
		
		// update in, out vector
		// hierarchical softmax
		if (mode == Mode::hierarchicalSoftmax)
		{
			for (auto j = jBegin; j < jEnd; ++j)
			{
				if (i == j) continue;
				float ll = inplaceUpdate(x, ws[j], z, lr1);
				assert(isnormal(ll));
				totalLLCnt++;
				totalLL += (ll - totalLL) / totalLLCnt;
			}
		}
		// negative sampling
		else
		{
			for (auto j = jBegin; j < jEnd; ++j)
			{
				if (i == j) continue;
				float ll = inplaceUpdate_NS(x, ws[j], z, lr1, false);
				assert(isnormal(ll));
				totalLLCnt++;
				totalLL += (ll - totalLL) / totalLLCnt;
			}
			for (auto ns : negativeSamples)
			{
				float ll = inplaceUpdate_NS(x, ns, z, lr1, true);
				assert(isnormal(ll));
				totalLL += ll / totalLLCnt;
			}
		}

		// variational update for q(pi_v)
		updateCounts(x, z, lr2);

		procWords += 1;

		if (threadId == 0 && procWords % 10000 == 0)
		{
			float time_per_kword = (procWords - lastProcWords) / timer.getElapsed() / 1000.f;
			printf("%.2f%% %.4f %.4f %.4f %.2f/%d %.2f kwords/sec\n",
				procWords / (totalWords / 100.f), totalLL, lr1, lr2,
				(float)senses / (i + 1), max_senses, time_per_kword);
			lastProcWords = procWords;
			timer.reset();
		}
	}
}

void TimeGramModel::updateNormalizedVector()
{
	inNormalized = MatrixXf::Zero(in.rows(), in.cols());
	for (size_t i = 0; i < T * vocabs.size(); ++i)
	{
		inNormalized.col(i) = in.col(i).normalized();
	}
}

void TimeGramModel::train(const function<vector<string>(size_t)>& reader, 
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
	vector<vector<uint32_t>> collections;
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
				futures.emplace_back(workers.enqueue([&d, &ld, window_length, start_lr, this](size_t threadId)
				{
					trainVectors(d.data(), d.size(), window_length, start_lr, ld[threadId], threadId);
				}));
			}
			for (auto& f : futures) f.get();
		}
		else
		{
			for (auto& d : collections)
			{
				trainVectors(d.data(), d.size(), window_length, start_lr, globalData);
			}
		}
		collections.clear();
	};

	for (size_t e = 0; e < epoch; ++e)
	{
		for (size_t id = 0; ; ++id)
		{
			auto rdoc = reader(id);
			if (rdoc.empty()) break;

			vector<uint32_t> doc;
			doc.reserve(rdoc.size());
			for (auto& w : rdoc)
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

			collections.emplace_back(move(doc));
			if (collections.size() >= batch)
			{
				procCollection();
			}
		}
	}
	procCollection();

	updateNormalizedVector();
}

void TimeGramModel::buildTrain(istream & is, size_t minCnt, 
	const function<bool(const string&)>& test, const function<string(const string&)>& trans,
	size_t numWorkers, size_t window_length, float start_lr, size_t batchSents, size_t epochs)
{
	istream_iterator<string> iBegin{ is }, iEnd{};
	buildVocab(iBegin, iEnd, minCnt, test, trans);
	train([&is, &trans](size_t id)->vector<string>
	{
		if (id == 0)
		{
			is.clear();
			is.seekg(0);
		}
		string line;
		while(1)
		{
			if (!getline(is, line)) return {};
			istringstream iss{ line };
			istream_iterator<string> iBegin{ iss }, iEnd{};
			vector<string> ret;
			transform(iBegin, iEnd, back_inserter(ret), trans);
			if (ret.empty()) continue;
			return move(ret);
		}
	}, numWorkers, window_length, start_lr, batchSents, epochs);
}

pair<VectorXf, size_t> TimeGramModel::getExpectedPi(const string & word) const
{
	size_t wv = vocabs.get(word);
	if (wv == (size_t)-1) return {};
	return getExpectedPi(wv);
}

vector<tuple<string, size_t, float>> TimeGramModel::nearestNeighbors(const string & word, size_t ws, size_t K, float min_count) const
{
	const size_t V = vocabs.size();
	size_t wv = vocabs.get(word);
	if (wv == (size_t)-1) return {};
	auto vec = inNormalized.col(wv * T + ws);

	if (counts(ws, wv) < min_count) return {};

	vector<tuple<string, size_t, float>> top;
	MatrixXf sim(T, V);
	for (size_t v = 0; v < V; ++v)
	{
		for (size_t s = 0; s < T; ++s)
		{
			if ((v == wv && s == ws) || counts(s, v) < min_count)
			{
				sim(s, v) = -INFINITY;
				continue;
			}
			sim(s, v) = inNormalized.col(v * T + s).dot(vec);
		}
	}

	for (size_t k = 0; k < K; ++k)
	{
		size_t idx = max_element(sim.data(), sim.data() + sim.size()) - sim.data();
		top.emplace_back(vocabs.getStr(idx / T), idx % T, sim.data()[idx]);
		sim.data()[idx] = -INFINITY;
	}
	return top;
}

vector<float> TimeGramModel::disambiguate(const string & word, const vector<string>& context, bool use_prior, float min_prob) const
{
	const size_t V = vocabs.size();
	size_t wv = vocabs.get(word);
	if (wv == (size_t)-1) return {};

	VectorXf z = VectorXf::Zero(T);
	if (use_prior)
	{
		z = getExpectedPi(wv).first;
		for (size_t k = 0; k < T; ++k)
		{
			if (z[k] < min_prob) z[k] = 0;
			z[k] = log(z[k]);
		}
	}

	for (auto& y : context)
	{
		size_t yId = vocabs.get(y);
		if (yId == (size_t)-1) continue;
		updateZ(z, wv, yId);
	}

	exp_normalize(z);
	return { z.data(), z.data() + T };
}

vector<tuple<string, size_t, float>> TimeGramModel::mostSimilar(const vector<pair<string, size_t>>& positiveWords, 
	const vector<pair<string, size_t>>& negativeWords, size_t K, float min_count) const
{
	VectorXf vec = VectorXf::Zero(M);
	const size_t V = vocabs.size();
	unordered_set<size_t> uniqs;
	for (auto& p : positiveWords)
	{
		size_t wv = vocabs.get(p.first);
		if (wv == (size_t)-1) return {};
		if (counts(p.second, wv) < min_count) return {};
		vec += inNormalized.col(wv * T + p.second);
		uniqs.emplace(wv * T + p.second);
	}

	for (auto& p : negativeWords)
	{
		size_t wv = vocabs.get(p.first);
		if (wv == (size_t)-1) return {};
		if (counts(p.second, wv) < min_count) return {};
		vec -= inNormalized.col(wv * T + p.second);
		uniqs.emplace(wv * T + p.second);
	}

	vec.normalize();

	vector<tuple<string, size_t, float>> top;
	MatrixXf sim(T, V);
	for (size_t v = 0; v < V; ++v)
	{
		for (size_t s = 0; s < T; ++s)
		{
			if (uniqs.count(v * T + s) || counts(s, v) < min_count)
			{
				sim(s, v) = -INFINITY;
				continue;
			}
			sim(s, v) = inNormalized.col(v * T + s).dot(vec);
		}
	}

	for (size_t k = 0; k < K; ++k)
	{
		size_t idx = max_element(sim.data(), sim.data() + sim.size()) - sim.data();
		top.emplace_back(vocabs.getStr(idx / T), idx % T, sim.data()[idx]);
		sim.data()[idx] = -INFINITY;
	}
	return top;
}

void TimeGramModel::saveModel(ostream & os) const
{
	writeToBinStream(os, (uint32_t)M);
	writeToBinStream(os, (uint32_t)T);
	writeToBinStream(os, alpha);
	writeToBinStream(os, d);
	writeToBinStream(os, sense_threshold);
	writeToBinStream(os, (uint32_t)context_cut);
	writeToBinStream(os, (uint32_t)code.rows());
	vocabs.writeToFile(os);
	writeToBinStream(os, frequencies);
	writeToBinStream(os, in);
	writeToBinStream(os, out);
	writeToBinStream(os, counts);
	writeToBinStream(os, code);
	writeToBinStream(os, path);
}

TimeGramModel TimeGramModel::loadModel(istream & is)
{
	size_t M = readFromBinStream<uint32_t>(is);
	size_t T = readFromBinStream<uint32_t>(is);
	float alpha = readFromBinStream<float>(is);
	float d = readFromBinStream<float>(is);
	float sense_threshold = readFromBinStream<float>(is);
	bool context_cut = readFromBinStream<uint32_t>(is);
	size_t max_length = readFromBinStream<uint32_t>(is);
	TimeGramModel ret{ M, T, alpha, d };
	ret.sense_threshold = sense_threshold;
	ret.context_cut = context_cut;
	ret.vocabs.readFromFile(is);
	size_t V = ret.vocabs.size();
	ret.in.resize(M, T*V);
	ret.out.resize(M, V);
	ret.counts.resize(T, V);
	ret.code.resize(max_length, V);
	ret.path.resize(max_length, V);

	readFromBinStream(is, ret.frequencies);
	readFromBinStream(is, ret.in);
	readFromBinStream(is, ret.out);
	readFromBinStream(is, ret.counts);
	readFromBinStream(is, ret.code);
	readFromBinStream(is, ret.path);

	ret.updateNormalizedVector();
	return ret;
}
