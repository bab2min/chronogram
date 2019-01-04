#pragma once

#include <unordered_map>
#include <random>
#include <functional>
#include <mutex>
#include <atomic>
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
			while (id >= rfreqs.size()) rfreqs.emplace_back(0);
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

	class LLEvaluater
	{
		friend class TimeGramModel;
		size_t L, negativeSampleSize, windowLen, nsQ;
		std::vector<uint32_t> wordIds;
		struct MixedVectorCoef
		{
			size_t n = 0;
			std::map<uint32_t, std::vector<float>> dataMap;
			std::vector<float> dataVec;

			const float* get(uint32_t id, size_t nsQ, size_t L) const
			{
				if(nsQ && id % nsQ == n) return &dataVec[id / nsQ * L];
				auto it = dataMap.find(id);
				if (it != dataMap.end()) return &it->second[0];
				return nullptr;
			}
		};

		std::unordered_map<uint32_t, MixedVectorCoef> coefs;
		const std::vector<float>& unigramDist;

		LLEvaluater(size_t _L, size_t _negativeSampleSize, size_t _windowLen,
			size_t _nsQ, std::vector<uint32_t>&& _wordIds, 
			std::unordered_map<uint32_t, MixedVectorCoef>&& _coefs,
			const std::vector<float>& _unigramDist)
			: L(_L), negativeSampleSize(_negativeSampleSize),
			windowLen(_windowLen), nsQ(_nsQ),
			wordIds(_wordIds), coefs(_coefs), unigramDist(_unigramDist)
		{}

	public:
		float operator()(float timePoint) const;
		std::tuple<float, float, float> fgh(float timePoint) const;
		std::tuple<float, float> gh(float timePoint) const;
	};

private:
	struct ThreadLocalData
	{
		std::mt19937_64 rg;
		Eigen::MatrixXf updateOutMat;
		std::unordered_map<uint32_t, uint32_t> updateOutIdx;
	};

	std::vector<size_t> frequencies; // (V)
	std::vector<size_t> wordProcd;
	Eigen::MatrixXf in; // (M, L * V)
	Eigen::MatrixXf out; // (M, V)
	Eigen::MatrixXf wordDist; // (L, V)
	size_t M; // dimension of word vector
	size_t L; // order of Lengendre polynomial
	float subsampling;
	float zBias = 0, zSlope = 1;
	float eta = 1.f;
	Mode mode;

	size_t totalWords = 0;
	size_t procWords = 0, lastProcWords = 0;
	size_t totalLLCnt = 0;
	double totalLL = 0;
	double avgWordDistErr = 0;

	ThreadLocalData globalData;
	WordDictionary<> vocabs;

	std::vector<float> unigramDist;
	std::discrete_distribution<uint32_t> unigramTable;
	size_t negativeSampleSize = 0;

	Timer timer;

	std::mutex mtx;

	static Eigen::VectorXf makeCoef(size_t L, float z);
	static Eigen::VectorXf makeDCoef(size_t L, float z);
	Eigen::VectorXf makeTimedVector(size_t wv, const Eigen::VectorXf& coef) const;

	float inplaceUpdate(size_t x, size_t y, float lr, bool negative, const Eigen::VectorXf& lWeight);
	float getUpdateGradient(size_t x, size_t y, float lr, bool negative, const Eigen::VectorXf& lWeight,
		Eigen::DenseBase<Eigen::MatrixXf>::ColXpr xGrad,
		Eigen::DenseBase<Eigen::MatrixXf>::ColXpr yGrad);
	void buildModel();
	void buildTable();
	void trainVectors(const uint32_t* ws, size_t N, float timePoint,
		size_t window_length, float start_lr, size_t nEpoch, size_t report);
	void trainVectorsMulti(const uint32_t* ws, size_t N, float timePoint,
		size_t window_length, float start_lr, size_t nEpoch, size_t report, ThreadLocalData& ld);
	void normalizeWordDist();

	float getWordProbByTime(uint32_t w, float timePoint) const;
public:
	TimeGramModel(size_t _M = 100, size_t _L = 6,
		float _subsampling = 1e-4, size_t _negativeSampleSize = 5, float _eta = 1.f,
		size_t seed = std::random_device()())
		: M(_M), L(_L), subsampling(_subsampling), eta(_eta),
		mode(_negativeSampleSize ? Mode::negativeSampling : Mode::hierarchicalSoftmax),
		negativeSampleSize(_negativeSampleSize)
	{
		globalData.rg = std::mt19937_64{ seed };
	}

	TimeGramModel(TimeGramModel&& o)
		: M(o.M), L(o.L), globalData(o.globalData),
		vocabs(std::move(o.vocabs)), frequencies(std::move(o.frequencies)), 
		unigramTable(std::move(o.unigramTable)), unigramDist(std::move(o.unigramDist)),
		in(std::move(o.in)), out(std::move(o.out)), zBias(o.zBias), zSlope(o.zSlope)
	{
	}

	TimeGramModel& operator=(TimeGramModel&& o)
	{
		M = o.M;
		L = o.L;
		globalData = o.globalData;
		vocabs = std::move(o.vocabs);
		frequencies = std::move(o.frequencies);
		unigramTable = std::move(o.unigramTable);
		unigramDist = std::move(o.unigramDist);
		in = std::move(o.in);
		out = std::move(o.out);
		zBias = o.zBias;
		zSlope = o.zSlope;
		return *this;
	}

	void buildVocab(const std::function<ReadResult(size_t)>& reader, size_t minCnt = 10);
	void train(const std::function<ReadResult(size_t)>& reader, size_t numWorkers = 0,
		size_t window_length = 4, float start_lr = 0.025, size_t batchSents = 1000, size_t epochs = 1, size_t report = 10000);

	float arcLengthOfWord(const std::string& word, size_t step = 100) const;
	std::vector<std::tuple<std::string, float>> nearestNeighbors(const std::string& word, 
		float wordTimePoint, float searchingTimePoint, size_t K = 10) const;
	std::vector<std::tuple<std::string, float>> mostSimilar(
		const std::vector<std::pair<std::string, float>>& positiveWords,
		const std::vector<std::pair<std::string, float>>& negativeWords,
		float searchingTimePoint, size_t K = 10) const;

	LLEvaluater evaluateSent(const std::vector<std::string>& words, size_t windowLen, size_t nsQ = 16) const;
	std::pair<float, float> predictSentTime(const std::vector<std::string>& words, size_t windowLen, size_t nsQ = 16, size_t initStep = 24) const;

	const std::vector<std::string>& getVocabs() const
	{
		return vocabs.getKeys();
	}

	Eigen::MatrixXf getEmbedding(const std::string& word) const;

	void saveModel(std::ostream& os) const;
	static TimeGramModel loadModel(std::istream& is);

	float getMinPoint() const { return zBias; }
	float getMaxPoint() const { return zBias + 1 / zSlope; }
	float normalizedTimePoint(float t) const { return (t - zBias) * zSlope; }
	float unnormalizedTimePoint(float t) const { return t / zSlope + zBias; }

	float getWordProbByTime(const std::string& word, float timePoint) const;
};

