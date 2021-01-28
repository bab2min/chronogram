#pragma once

#include <unordered_map>
#include <unordered_set>
#include <random>
#include <functional>
#include <mutex>
#include <atomic>
#include <Eigen/Dense>
#include <EigenRand/EigenRand>
#include "ThreadPool.h"
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

namespace act
{
	struct Linear
	{
		std::pair<float, float> operator()(float x)
		{
			return std::make_pair(x, 1);
		}
	};

	struct Sqrt
	{
		std::pair<float, float> operator()(float x)
		{
			x = std::sqrt(x);
			return std::make_pair(x, 0.5f / x);
		}
	};

	struct Tanh
	{
		std::pair<float, float> operator()(float x)
		{
			x = std::tanh(x);
			return std::make_pair(x, 1 - x * x);
		}
	};
}

class ChronoGramModel
{
public:
	using TempAct = act::Linear;

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

		struct MixedVectorCoef
		{
			size_t n = 0;
			std::map<uint32_t, std::vector<float>> dataMap;
			std::vector<float> dataVec;

			const float* get(uint32_t id, size_t nsQ, size_t R) const
			{
				if (nsQ && id % nsQ == n) return &dataVec[id / nsQ * R];
				auto it = dataMap.find(id);
				if (it != dataMap.end()) return &it->second[0];
				return nullptr;
			}
		};

		const ChronoGramModel& tgm;
		size_t windowLen, nsQ;
		float timePriorWeight;

		std::vector<uint32_t> wordIds;
		std::unordered_map<uint32_t, MixedVectorCoef> coefs;
		std::unordered_map<uint32_t, Eigen::MatrixXf> ugCoefs;
		std::vector<std::vector<uint32_t>> subwordTables;
		std::function<float(float)> timePrior;

		LLEvaluater(const ChronoGramModel& _tgm, size_t _windowLen = 0,
			size_t _nsQ = 0, float _timePriorWeight = 0)
			: tgm(_tgm), windowLen(_windowLen), nsQ(_nsQ), timePriorWeight(_timePriorWeight)
		{}

	public:
		LLEvaluater(const LLEvaluater&) = default;
		LLEvaluater(LLEvaluater&&) = default;

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

	struct alignas(4) HyperParameter
	{
		uint32_t dimension = 100;
		uint32_t order = 6;
		float subsampling = 1e-4f;
		float temporalSubsampling = 1;
		uint32_t negativeSamples = 5;
		uint32_t temporalNegativeSamples = 5;
		float eta = 1;
		float zeta = .1f;
		float lambda = .1f;
		uint32_t weightDecayInterval = 0;
		float orderDecay = 0;
		float weightDecay = 0;
		uint32_t subwordGrams = 0;
		uint32_t subwordDropoutRemain = 5;
		float subwordWeightDecay = 0;
		float tnsWeight = 0;
		float ugWeight = 0;
		float dropout = 0;
	};

private:
	struct ThreadLocalData
	{
		std::mt19937_64 rg;
		Eigen::Rand::Vmt19937_64 vrg;
		Eigen::MatrixXf updateOutMat;
		std::unordered_map<uint32_t, uint32_t> updateOutIdx;
		std::unordered_set<uint32_t> updateOutIdxHash;
	};

	struct TrainResult
	{
		size_t numWords = 0, numPairs = 0;
		float contextLL = 0, temporalLL = 0, ugLL = 0;
	};

	std::vector<uint64_t> frequencies; // (V)
	std::vector<float> wordScale;
	std::unordered_set<uint32_t> fixedWords;
	Eigen::MatrixXf in; // (D, R * V)
	Eigen::MatrixXf subwordIn; // (D, R * SV)
	Eigen::MatrixXf out; // (D, V)
	Eigen::MatrixXf ugOut; // (D, V)
	Eigen::VectorXf ugCoef, subwordUgCoef;

	HyperParameter hp;
	float zBias = 0, zSlope = 1;

	float timePadding = 0;
	float timePriorScale = 1;
	Eigen::VectorXf timePrior; // (R, 1)
	Eigen::VectorXf vEta;

	float tpvThreshold = 0.125f, tpvBias = 0.03125f;

	size_t totalWords = 0, totalTimePoints = 0;
	size_t procWords = 0, lastProcWords = 0, procTimePoints = 0;
	size_t timeLLCnt = 0;
	double timeLL = 0;

	ThreadLocalData globalData;
	WordDictionary<> vocabs, subwordVocabs;

	std::vector<float> unigramDist;
	std::discrete_distribution<uint32_t> unigramTable;
	std::vector<uint32_t> subwordTable;
	std::vector<size_t> subwordTablePtrs;
	Eigen::ArrayXf inDecay, outDecay, subwordInDecay;
	Eigen::Rand::BernoulliGen<float> dropoutGen;

	Timer timer;

	static Eigen::VectorXf makeCoef(size_t R, float z);
	static Eigen::VectorXf makeDCoef(size_t R, float z);
	const Eigen::VectorXf& makeTimedVector(size_t wv, const Eigen::VectorXf& coef) const;
	const Eigen::VectorXf& makeTimedVectorDropout(size_t wv, const Eigen::VectorXf& coef, const std::vector<uint32_t>& subwords) const;
	const Eigen::VectorXf& makeSubwordTimedVector(const std::vector<uint32_t>& swv, const Eigen::VectorXf& coef) const;

	std::vector<uint32_t> getSubwordIds(const std::string& s) const;

	template<bool _initialization = false, bool _use_ug = false>
	float inplaceUpdate(size_t x, size_t y, float lr, bool negative, 
		const Eigen::VectorXf& lWeight, const std::vector<uint32_t>& subwords
	);

	template<bool _initialization = false, bool _use_ug = false>
	float getUpdateGradient(size_t x, size_t y, float lr, bool negative, 
		const Eigen::VectorXf& lWeight, const std::vector<uint32_t>& subwords,
		Eigen::DenseBase<Eigen::MatrixXf>::ColXpr xGrad,
		Eigen::DenseBase<Eigen::MatrixXf>::ColXpr yGrad,
		ThreadLocalData& ld
	);

	float inplaceTimeUpdate(size_t x, float lr, const Eigen::VectorXf& lWeight, const std::vector<uint32_t>& subwords,
		const std::vector<const Eigen::VectorXf*>& randSampleWeight);
	float getTimeUpdateGradient(size_t x, float lr, const Eigen::VectorXf& lWeight, const std::vector<uint32_t>& subwords,
		const std::vector<const Eigen::VectorXf*>& randSampleWeight,
		Eigen::Block<Eigen::MatrixXf> grad);

	float inplaceTimeUpdateV2(size_t x, float lr, const Eigen::VectorXf& lWeight, const std::vector<uint32_t>& subwords,
		const std::vector<const Eigen::VectorXf*>& randSampleWeight);

	template<typename _ActFn>
	float getTimeUpdateGradientV2(size_t x, float lr, const Eigen::VectorXf& lWeight, const std::vector<uint32_t>& subwords,
		const std::vector<Eigen::VectorXf>& randSampleWeight,
		Eigen::MatrixXf& embGrad, float& coefGrad, std::vector<float>& subwordCoefGrad, _ActFn&& act);

	float updateTimePrior(float lr, const Eigen::VectorXf& lWeight, const std::vector<const Eigen::VectorXf*>& randSampleWeight);

	void buildModel();
	void buildTable();
	void buildSubwordTable();

	template<bool _multi>
	void updateWeightDecay(float lr, std::mutex* mtxIn, std::mutex* mtxSubwordIn, std::mutex* mtxOut, size_t hashSize);

	template<bool _Initialization = false, bool _GNgramMode = false>
	TrainResult trainVectors(const uint32_t* ws, size_t N, float timePoint,
		size_t windowLen, float lr);

	template<bool _Initialization = false, bool _GNgramMode = false>
	TrainResult trainVectorsMulti(const uint32_t* ws, size_t N, float timePoint,
		size_t windowLen, float lr, ThreadLocalData& ld,
		std::mutex* mtxIn, std::mutex* mtxSubwordIn, std::mutex* mtxOut, size_t hashSize);

	void trainTimePrior(const float* ts, size_t N, float lr, size_t report);
	void normalizeWordDist(bool updateVocab = true, ThreadPool* pool = nullptr);

	float getTimePrior(const Eigen::VectorXf& coef) const;
	float getWordProbByTime(uint32_t w, const Eigen::VectorXf& timedVector, const Eigen::VectorXf& coef, float tPrior) const;
	float getWordProbByTime(uint32_t w, float timePoint) const;
public:
	ChronoGramModel()
	{
	}

	ChronoGramModel(const HyperParameter& _hp, size_t seed = std::random_device()())
		: hp(_hp), vEta(Eigen::VectorXf::Constant(_hp.order, _hp.eta))
	{
		globalData.rg = std::mt19937_64{ seed };
		globalData.vrg = Eigen::Rand::Vmt19937_64{ seed };

		vEta[0] = 1;
		timePadding = .25f / _hp.order;
	}

	ChronoGramModel(ChronoGramModel&& o) = default;

	ChronoGramModel& operator=(ChronoGramModel&& o) = default;

	void buildVocab(const std::function<ResultReader()>& reader, size_t minCnt = 10, size_t minCntForSubword = 5, size_t numWorkers = 0);
	size_t recountVocab(const std::function<ResultReader()>& reader, float minT, float maxT, size_t numWorkers);
	size_t recountVocab(const std::function<GNgramResultReader()>& reader, float minT, float maxT, size_t numWorkers);
	bool addFixedWord(const std::string& word);

	void buildVocabFromDict(const std::function<std::pair<std::string, uint64_t>()>& reader, float minT, float maxT, size_t vocabSize = -1);

	using ReportCallback = std::function<bool(size_t, float, float, float, float, float, float)>;

	static bool defaultReportCallback(size_t steps, float progress, float contextLL, float temporalLL, float ugLL, float lr, float timePerKword);

	template<bool _Initialization = false>
	void train(const std::function<ResultReader()>& reader, size_t numWorkers = 0,
		size_t windowLen = 4, float start_lr = .025f, float end_lr = .00025f, size_t batchSents = 1000, float epochs = 1, size_t report = 100000,
		const ReportCallback& reportCallback = defaultReportCallback);

	template<bool _Initialization = false>
	void trainFromGNgram(const std::function<GNgramResultReader()>& reader, uint64_t maxItems, size_t numWorkers = 0,
		float start_lr = .025f, float end_lr = .00025f, size_t batchSents = 1000, float epochs = 1, size_t report = 100000,
		const ReportCallback& reportCallback = defaultReportCallback);

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
		size_t nsQ = 16, const std::function<float(float)>& timePrior = {}, float timePriorWeight = 0,
		bool estimateSubword = false) const;

	std::pair<float, float> predictSentTime(const std::vector<std::string>& words, 
		size_t windowLen, size_t nsQ = 16, const std::function<float(float)>& timePrior = {}, float timePriorWeight = 0,
		size_t initStep = 8, float threshold = .0025f, std::vector<float>* llOutput = nullptr) const;

	std::vector<EvalResult> evaluate(const std::function<ReadResult()>& reader, 
		const std::function<void(EvalResult)>& writer,
		size_t numWorkers, size_t windowLen, size_t nsQ, const std::function<float(float)>& timePrior, float timePriorWeight,
		size_t initStep, float threshold) const;

	size_t usedVocabSize() const { return frequencies.size(); }

	const std::vector<std::string>& getVocabs() const
	{
		return vocabs.getKeys();
	}

	const std::vector<std::string>& getSubwordVocabs() const
	{
		return subwordVocabs.getKeys();
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

	const HyperParameter& getHP() const { return hp; }

	size_t getTotalWords() const;
	size_t getWordCount(const std::string& word) const;

	void setPadding(float padding) { timePadding = padding; }
	float getPadding() const { return timePadding; }

	void setTPBias(float _tpvBias) { tpvBias = _tpvBias; }
	float getTPBias() const { return tpvBias; }
	void setTPThreshold(float _tpvThreshold) { tpvThreshold = _tpvThreshold; }
	float getTPThreshold() const { return tpvThreshold; }
};


template<typename _Ty, int _Rows, int _Cols>
struct Serializer<Eigen::Matrix<_Ty, _Rows, _Cols>>
{
	template<typename _Os>
	void write(_Os&& os, const Eigen::Matrix<_Ty, _Rows, _Cols>& v)
	{
		for (size_t i = 0; i < v.size(); ++i)
		{
			writeToBinStream(os, v.data()[i]);
		}
	}

	template<typename _Is>
	void read(_Is&& is, Eigen::Matrix<_Ty, _Rows, _Cols>& v)
	{
		for (size_t i = 0; i < v.size(); ++i)
		{
			readFromBinStream(is, v.data()[i]);
		}
	}
};

template<>
struct Serializer<ChronoGramModel::HyperParameter>
{
	template<typename _Os>
	void write(_Os&& os, const ChronoGramModel::HyperParameter& v)
	{
		writeToBinStream(os, (uint32_t)sizeof(v));
		os.write((const char*)&v, sizeof(v));
	}

	template<typename _Is>
	void read(_Is&& is, ChronoGramModel::HyperParameter& v)
	{
		size_t s = readFromBinStream<uint32_t>(is);
		v = {};
		is.read((char*)&v, s);
	}
};

template<bool _lock>
class optional_lock
{
public:
	optional_lock(const std::mutex&) {}
};

template<>
class optional_lock<true> : public std::lock_guard<std::mutex>
{
public:
	using std::lock_guard<std::mutex>::lock_guard;
};

