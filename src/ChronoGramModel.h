#pragma once

#include <unordered_map>
#include <unordered_set>
#include <random>
#include <functional>
#include <mutex>
#include <atomic>
#include <Eigen/Dense>
#include "dictionary.h"
#include "mathUtils.h"
#include "Timer.h"
#include "IOUtils.h"

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

class ChronoGramModel
{
public:
	struct ReadResult
	{
		std::vector<std::string> words;
		float timePoint = 0;
		bool stop = false;
	};
	using ResultReader = std::function<ReadResult()>;

	struct GNgramReadResult
	{
		std::array<uint32_t, 5> ngram;
		std::vector<std::pair<float, uint32_t>> yearCnt;
	};
	using GNgramResultReader = std::function<GNgramReadResult()>;

	class LLEvaluater
	{
		friend class ChronoGramModel;
		float timePriorWeight;
		size_t windowLen, nsQ;
		std::vector<uint32_t> wordIds;
		struct MixedVectorCoef
		{
			size_t n = 0;
			std::map<uint32_t, std::vector<float>> dataMap;
			std::vector<float> dataVec;

			const float* get(uint32_t id, size_t nsQ, size_t R) const
			{
				if(nsQ && id % nsQ == n) return &dataVec[id / nsQ * R];
				auto it = dataMap.find(id);
				if (it != dataMap.end()) return &it->second[0];
				return nullptr;
			}
		};

		std::unordered_map<uint32_t, MixedVectorCoef> coefs;
		const ChronoGramModel& tgm;
		std::function<float(float)> timePrior;
		LLEvaluater(const ChronoGramModel& _tgm, size_t _windowLen,
			size_t _nsQ, std::vector<uint32_t>&& _wordIds, 
			std::unordered_map<uint32_t, MixedVectorCoef>&& _coefs,
			const std::function<float(float)>& _timePrior,
			float _timePriorWeight)
			: windowLen(_windowLen), nsQ(_nsQ),
			wordIds(_wordIds), coefs(_coefs), tgm(_tgm),
			timePrior(_timePrior), timePriorWeight(_timePriorWeight)
		{}

	public:
		float operator()(float normalizedTimePoint) const;
		std::tuple<float, float> fg(float normalizedTimePoint) const;
	};

	struct EvalResult
	{
		float trueTime;
		float estimatedTime;
		float ll;
		float llPerWord;
		float normalizedErr;
		std::vector<std::string> words;
		std::vector<float> lls;
	};

private:
	struct ThreadLocalData
	{
		std::mt19937_64 rg;
		Eigen::MatrixXf updateOutMat;
		std::unordered_map<uint32_t, uint32_t> updateOutIdx;
		std::unordered_set<uint32_t> updateOutIdxHash;
	};

	struct TrainResult
	{
		size_t numWords = 0, numPairs = 0;
		float ll = 0, llUg = 0;
	};

	std::vector<uint64_t> frequencies; // (V)
	std::vector<float> wordScale;
	std::unordered_set<uint32_t> fixedWords;
	Eigen::MatrixXf in; // (D, R * V)
	Eigen::MatrixXf out; // (D, V)
	size_t D; // dimension of word vector
	size_t R; // order of Lengendre polynomial
	float subsampling;
	float zBias = 0, zSlope = 1;
	float zeta = .5f, lambda = .1f;

	float timePadding = 0;
	float timePriorScale = 1;
	Eigen::VectorXf timePrior; // (R, 1)
	Eigen::VectorXf vEta;

	float tpvThreshold = 0.25f, tpvBias = 0.0625f;

	size_t totalWords = 0, totalTimePoints = 0;
	size_t procWords = 0, lastProcWords = 0, procTimePoints = 0;
	size_t totalLLCnt = 0, timeLLCnt = 0;
	double totalLL = 0, ugLL = 0, timeLL = 0;

	ThreadLocalData globalData;
	WordDictionary<> vocabs;

	std::vector<float> unigramDist;
	std::discrete_distribution<uint32_t> unigramTable;
	size_t negativeSampleSize = 0, temporalSample = 0;

	Timer timer;

	static Eigen::VectorXf makeCoef(size_t R, float z);
	static Eigen::VectorXf makeDCoef(size_t R, float z);
	Eigen::VectorXf makeTimedVector(size_t wv, const Eigen::VectorXf& coef) const;

	template<bool _Initialization = false>
	float inplaceUpdate(size_t x, size_t y, float lr, bool negative, const Eigen::VectorXf& lWeight);

	template<bool _Initialization = false>
	float getUpdateGradient(size_t x, size_t y, float lr, bool negative, const Eigen::VectorXf& lWeight,
		Eigen::DenseBase<Eigen::MatrixXf>::ColXpr xGrad,
		Eigen::DenseBase<Eigen::MatrixXf>::ColXpr yGrad);

	float inplaceTimeUpdate(size_t x, float lr, const Eigen::VectorXf& lWeight, 
		const std::vector<const Eigen::VectorXf*>& randSampleWeight);
	float getTimeUpdateGradient(size_t x, float lr, const Eigen::VectorXf& lWeight,
		const std::vector<const Eigen::VectorXf*>& randSampleWeight,
		Eigen::Block<Eigen::MatrixXf> grad);

	float updateTimePrior(float lr, const Eigen::VectorXf& lWeight, const std::vector<const Eigen::VectorXf*>& randSampleWeight);

	void buildModel();
	void buildTable();

	template<bool _Initialization = false, bool _GNgramMode = false>
	TrainResult trainVectors(const uint32_t* ws, size_t N, float timePoint,
		size_t windowLen, float lr);

	template<bool _Initialization = false, bool _GNgramMode = false>
	TrainResult trainVectorsMulti(const uint32_t* ws, size_t N, float timePoint,
		size_t windowLen, float lr, ThreadLocalData& ld,
		std::mutex* mtxIn, std::mutex* mtxOut, size_t numWorkers = 1);
	void trainTimePrior(const float* ts, size_t N, float lr, size_t report);
	void normalizeWordDist(bool updateVocab = true);

	float getTimePrior(const Eigen::VectorXf& coef) const;
	float getWordProbByTime(uint32_t w, const Eigen::VectorXf& timedVector, const Eigen::VectorXf& coef, float tPrior) const;
	float getWordProbByTime(uint32_t w, float timePoint) const;
public:
	ChronoGramModel(size_t _D = 100, size_t _R = 6,
		float _subsampling = 1e-4f, size_t _negativeSampleSize = 5, size_t _timeNegativeSample = 5,
		float _eta = 1.f, float _zeta = .1f, float _lambda = .1f,
		size_t seed = std::random_device()())
		: D(_D), R(_R), subsampling(_subsampling), zeta(_zeta), lambda(_lambda),
		vEta(Eigen::VectorXf::Constant(_R, _eta)),
		negativeSampleSize(_negativeSampleSize), temporalSample(_timeNegativeSample)
	{
		globalData.rg = std::mt19937_64{ seed };

		vEta[0] = 1;
		timePadding = .25f / R;
	}

	ChronoGramModel(ChronoGramModel&& o)
		: D(o.D), R(o.R), globalData(o.globalData),
		vocabs(std::move(o.vocabs)), frequencies(std::move(o.frequencies)), 
		unigramTable(std::move(o.unigramTable)), unigramDist(std::move(o.unigramDist)),
		in(std::move(o.in)), out(std::move(o.out)), zBias(o.zBias), zSlope(o.zSlope),
		vEta(std::move(o.vEta)), zeta(o.zeta), lambda(o.lambda),
		timePrior(std::move(o.timePrior)), timePadding(o.timePadding),
		timePriorScale(o.timePriorScale), wordScale(std::move(o.wordScale)),
		negativeSampleSize(o.negativeSampleSize), temporalSample(o.temporalSample)
	{
	}

	ChronoGramModel& operator=(ChronoGramModel&& o)
	{
		D = o.D;
		R = o.R;
		globalData = o.globalData;
		vocabs = std::move(o.vocabs);
		frequencies = std::move(o.frequencies);
		unigramTable = std::move(o.unigramTable);
		unigramDist = std::move(o.unigramDist);
		in = std::move(o.in);
		out = std::move(o.out);
		zBias = o.zBias;
		zSlope = o.zSlope;
		vEta = std::move(o.vEta);
		zeta = o.zeta;
		lambda = o.lambda;
		timePrior = std::move(o.timePrior);
		timePadding = o.timePadding;
		timePriorScale = o.timePriorScale;
		wordScale = std::move(o.wordScale);
		negativeSampleSize = o.negativeSampleSize;
		temporalSample = o.temporalSample;
		return *this;
	}

	void buildVocab(const std::function<ResultReader()>& reader, size_t minCnt = 10, size_t numWorkers = 0);
	size_t recountVocab(const std::function<ResultReader()>& reader, float minT, float maxT, size_t numWorkers);
	size_t recountVocab(const std::function<GNgramResultReader()>& reader, float minT, float maxT, size_t numWorkers);
	bool addFixedWord(const std::string& word);

	void buildVocabFromDict(const std::function<std::pair<std::string, uint64_t>()>& reader, float minT, float maxT);

	template<bool _Initialization = false>
	void train(const std::function<ResultReader()>& reader, size_t numWorkers = 0,
		size_t windowLen = 4, float start_lr = .025f, float end_lr = .00025f, size_t batchSents = 1000, float epochs = 1, size_t report = 10000);

	template<bool _Initialization = false>
	void trainFromGNgram(const std::function<GNgramResultReader()>& reader, uint64_t maxItems, size_t numWorkers = 0,
		float start_lr = .025f, float end_lr = .00025f, size_t batchSents = 1000, float epochs = 1, size_t report = 10000);

	float arcLengthOfWord(const std::string& word, size_t step = 100) const;
	float angleOfWord(const std::string& word, size_t step = 100) const;

	std::vector<std::tuple<std::string, float, float>> nearestNeighbors(const std::string& word, 
		float wordTimePoint, float searchingTimePoint, float m = 0, size_t K = 10) const;
	std::vector<std::tuple<std::string, float, float>> mostSimilar(
		const std::vector<std::pair<std::string, float>>& positiveWords,
		const std::vector<std::pair<std::string, float>>& negativeWords,
		float searchingTimePoint, float m = 0, size_t K = 10, bool normalize = false) const;
	std::vector<std::tuple<std::string, float>> mostSimilarStatic(
		const std::vector<std::string>& positiveWords,
		const std::vector<std::string>& negativeWords,
		size_t K = 10) const;
	std::vector<std::pair<std::string, float>> calcShift(size_t minCnt, float time1, float time2, float m = 0) const;
	float sumSimilarity(const std::string& src, const std::vector<std::string>& targets, float timePoint, float m) const;
	float similarity(const std::string& word1, float time1, const std::string& word2, float time2) const;
	float similarityStatic(const std::string& word1, const std::string& word2) const;

	LLEvaluater evaluateSent(const std::vector<std::string>& words, size_t windowLen,
		size_t nsQ = 16, const std::function<float(float)>& timePrior = {}, float timePriorWeight = 0) const;
	std::pair<float, float> predictSentTime(const std::vector<std::string>& words, 
		size_t windowLen, size_t nsQ = 16, const std::function<float(float)>& timePrior = {}, float timePriorWeight = 0,
		size_t initStep = 8, float threshold = .0025f, std::vector<float>* llOutput = nullptr) const;

	std::vector<EvalResult> evaluate(const std::function<ReadResult()>& reader, 
		const std::function<void(EvalResult)>& writer,
		size_t numWorkers, size_t windowLen, size_t nsQ, const std::function<float(float)>& timePrior, float timePriorWeight,
		size_t initStep, float threshold) const;

	const std::vector<std::string>& getVocabs() const
	{
		return vocabs.getKeys();
	}

	Eigen::MatrixXf getEmbeddingMatrix(const std::string& word) const;
	Eigen::VectorXf getEmbedding(const std::string& word) const;
	Eigen::VectorXf getEmbedding(const std::string& word, float timePoint) const;

	void saveModel(std::ostream& os, bool compressed = true) const;
	template<class _Istream>
	static ChronoGramModel loadModel(_Istream& is);

	float getMinPoint() const { return zBias; }
	float getMaxPoint() const { return zBias + 1 / zSlope; }
	float normalizedTimePoint(float t) const { return (t - zBias) * zSlope * (1 - timePadding * 2) + timePadding; }
	float unnormalizedTimePoint(float t) const { return (t - timePadding) / (zSlope * (1 - timePadding * 2)) + zBias; }

	float getWordProbByTime(const std::string& word, float timePoint) const;
	float getTimePrior(float timePoint) const;

	size_t getR() const { return R; }
	size_t getD() const { return D; }

	size_t getTotalWords() const;
	size_t getWordCount(const std::string& word) const;

	float getZeta() const { return zeta; }
	float getLambda() const { return lambda; }

	void setPadding(float padding) { timePadding = padding; }
	float getPadding() const { return timePadding; }

	void setTPBias(float _tpvBias) { tpvBias = _tpvBias; }
	float getTPBias() const { return tpvBias; }
	void setTPThreshold(float _tpvThreshold) { tpvThreshold = _tpvThreshold; }
	float getTPThreshold() const { return tpvThreshold; }
};
