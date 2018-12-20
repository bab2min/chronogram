#pragma once

#include <unordered_map>
#include <random>
#include <functional>
#include <mutex>
#include <Eigen/Dense>
#include "dictionary.h"
#include "Timer.h"

struct VocabCounter
{
	WordDictionary<> rdict;
	std::vector<size_t> rfreqs;

	static bool defaultTest(const std::string& o) { return true; }
	static std::string defaultTrans(const std::string& o) { return o; }

	template<class _Iter, class _Pred, class _Transform>
	void update(_Iter begin, _Iter end,
		_Pred test, _Transform trans)
	{
		std::string word;
		for (; begin != end; ++begin)
		{
			if (!test(*begin)) continue;
			size_t id = rdict.getOrAdd(trans(*begin));
			if (id >= rfreqs.size()) rfreqs.resize(id + 1);
			rfreqs[id]++;
		}
	}
};

class TimeGramModel
{
public:
	enum class Mode
	{
		hierarchicalSoftmax,
		negativeSampling
	};

	struct ReadResult
	{
		std::vector<std::string> words;
		float timePoint = 0;
		bool stop = false;
	};

private:
	struct ThreadLocalData
	{
		std::mt19937_64 rg;
		Eigen::MatrixXf updateOutMat;
		std::unordered_map<uint32_t, uint32_t> updateOutIdx;
	};

	std::vector<size_t> frequencies; // (V)
	Eigen::MatrixXf in; // (M, L * V)
	Eigen::MatrixXf out; // (M, V)
	size_t M; // dimension of word vector
	size_t L; // order of Lengendre polynomial
	float subsampling;
	float zBias = 0, zSlope = 1;
	Mode mode;

	size_t totalWords = 0;
	size_t procWords = 0, lastProcWords = 0;
	size_t totalLLCnt = 0;
	double totalLL = 0;

	ThreadLocalData globalData;
	WordDictionary<> vocabs;

	std::discrete_distribution<uint32_t> unigramTable;
	size_t negativeSampleSize = 0;

	Timer timer;

	std::mutex mtx;

	static std::vector<float> makeCoef(size_t L, float z);
	Eigen::VectorXf makeTimedVector(size_t wv, const std::vector<float>& coef) const;

	float inplaceUpdate(size_t x, size_t y, float lr, bool negative, const std::vector<float>& lWeight);
	float getUpdateGradient(size_t x, size_t y, float lr, bool negative, const std::vector<float>& lWeight,
		Eigen::DenseBase<Eigen::MatrixXf>::ColXpr xGrad,
		Eigen::DenseBase<Eigen::MatrixXf>::ColXpr yGrad);
	void buildModel();
	void trainVectors(const uint32_t* ws, size_t N, float timePoint,
		size_t window_length, float start_lr);
	void trainVectorsMulti(const uint32_t* ws, size_t N, float timePoint,
		size_t window_length, float start_lr, ThreadLocalData& ld);
public:
	TimeGramModel(size_t _M = 100, size_t _L = 6,
		float _subsampling = 1e-4, size_t _negativeSampleSize = 5, size_t seed = std::random_device()())
		: M(_M), L(_L), subsampling(_subsampling),
		mode(_negativeSampleSize ? Mode::negativeSampling : Mode::hierarchicalSoftmax),
		negativeSampleSize(_negativeSampleSize)
	{
		globalData.rg = std::mt19937_64{ seed };
	}

	TimeGramModel(TimeGramModel&& o)
		: M(o.M), L(o.L), globalData(o.globalData),
		vocabs(std::move(o.vocabs)), frequencies(std::move(o.frequencies)),
		in(std::move(o.in)), out(std::move(o.out))
	{
	}

	TimeGramModel& operator=(TimeGramModel&& o)
	{
		M = o.M;
		L = o.L;
		globalData = o.globalData;
		vocabs = std::move(o.vocabs);
		frequencies = std::move(o.frequencies);
		in = std::move(o.in);
		out = std::move(o.out);
		return *this;
	}

	void buildVocab(const std::function<ReadResult(size_t)>& reader, size_t minCnt = 10);
	void train(const std::function<ReadResult(size_t)>& reader, size_t numWorkers = 0,
		size_t window_length = 4, float start_lr = 0.025, size_t batchSents = 1000, size_t epochs = 1);

	std::vector<std::tuple<std::string, float>> nearestNeighbors(const std::string& word, 
		float timePoint, size_t K = 10) const;
	std::vector<std::tuple<std::string, float>> mostSimilar(
		const std::vector<std::string>& positiveWords,
		const std::vector<std::string>& negativeWords,
		float timePoint, size_t K = 10) const;

	const std::vector<std::string>& getVocabs() const
	{
		return vocabs.getKeys();
	}

	void saveModel(std::ostream& os) const;
	static TimeGramModel loadModel(std::istream& is);

	float getMinPoint() const { return zBias; }
	float getMaxPoint() const { return zBias + 1 / zSlope; }
};

