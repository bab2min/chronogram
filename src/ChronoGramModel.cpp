#include <numeric>
#include <iostream>
#include <iterator>
#include "ChronoGramModel.h"
#include "polynomials.hpp"

using namespace std;
using namespace Eigen;

#define DEBUG_PRINT(out, x) (out << #x << ": " << x << endl)

void ChronoGramModel::buildModel()
{
	const size_t V = usedVocabSize();
	// allocate & initialize model
	in = MatrixXf::Zero(hp.dimension, hp.order * V);
	for (size_t v = 0; v < V; ++v)
	{
		in.block(0, v * hp.order, hp.dimension, 1) = MatrixXf::Random(hp.dimension, 1) * (.5f / hp.dimension);
	}

	out = MatrixXf::Random(hp.dimension, V) * (.5f / hp.dimension);

	timePrior = VectorXf::Zero(hp.order);
	timePrior[0] = 1;

	if (hp.dropout)
	{
		dropoutGen = Eigen::Rand::BernoulliGen<float>{ 1 - hp.dropout };
	}

	normalizeWordDist(false);
	buildTable();
	buildSubwordTable();
}

void ChronoGramModel::buildTable()
{
	// build Unigram Table for Negative Sampling
	vector<double> weights;
	transform(frequencies.begin(), frequencies.end(), back_inserter(weights), [](size_t w) { return pow(w, 0.75); });
	unigramTable = discrete_distribution<uint32_t>(weights.begin(), weights.end());
	auto p = unigramTable.probabilities();
	unigramDist = vector<float>{ p.begin(), p.end() };
	wordScale = vector<float>(usedVocabSize(), 1.f);
}

u32string decodeUTF8(const string& str)
{
	u32string ret;
	for (auto it = str.begin(); it != str.end(); ++it)
	{
		size_t code = 0;
		uint8_t byte = *it;
		if ((byte & 0xF8) == 0xF0)
		{
			code = (byte & 0x07) << 18;
			if (++it == str.end()) throw runtime_error{ "unexpected utf8 ending" };
			if (((byte = *it) & 0xC0) != 0x80) throw runtime_error{ "unexpected utf8 trailing byte" };
			code |= (byte & 0x3F) << 12;
			if (++it == str.end()) throw runtime_error{ "unexpected utf8 ending" };
			if (((byte = *it) & 0xC0) != 0x80) throw runtime_error{ "unexpected utf8 trailing byte" };
			code |= (byte & 0x3F) << 6;
			if (++it == str.end()) throw runtime_error{ "unexpected utf8 ending" };
			if (((byte = *it) & 0xC0) != 0x80) throw runtime_error{ "unexpected utf8 trailing byte" };
			code |= (byte & 0x3F);
		}
		else if ((byte & 0xF0) == 0xE0)
		{
			code = (byte & 0x0F) << 12;
			if (++it == str.end()) throw runtime_error{ "unexpected utf8 ending" };
			if (((byte = *it) & 0xC0) != 0x80) throw runtime_error{ "unexpected utf8 trailing byte" };
			code |= (byte & 0x3F) << 6;
			if (++it == str.end()) throw runtime_error{ "unexpected utf8 ending" };
			if (((byte = *it) & 0xC0) != 0x80) throw runtime_error{ "unexpected utf8 trailing byte" };
			code |= (byte & 0x3F);
		}
		else if ((byte & 0xE0) == 0xC0)
		{
			code = (byte & 0x1F) << 6;
			if (++it == str.end()) throw runtime_error{ "unexpected utf8 ending" };
			if (((byte = *it) & 0xC0) != 0x80) throw runtime_error{ "unexpected utf8 trailing byte" };
			code |= (byte & 0x3F);
		}
		else if ((byte & 0x80) == 0x00)
		{
			code = byte;
		}
		else
		{
			throw runtime_error{ "unicode error" };
		}
		ret.push_back(code);
	}
	return ret;
}

string encodeUTF8(const u32string& str)
{
	string ret;
	for (auto it = str.begin(); it != str.end(); ++it)
	{
		size_t code = *it;
		if (code <= 0x7F)
		{
			ret.push_back(code);
		}
		else if (code <= 0x7FF)
		{
			ret.push_back(0xC0 | (code >> 6));
			ret.push_back(0x80 | (code & 0x3F));
		}
		else if (code <= 0xFFFF)
		{
			ret.push_back(0xE0 | (code >> 12));
			ret.push_back(0x80 | ((code >> 6) & 0x3F));
			ret.push_back(0x80 | (code & 0x3F));
		}
		else if (code <= 0x10FFFF)
		{
			ret.push_back(0xF0 | (code >> 18));
			ret.push_back(0x80 | ((code >> 12) & 0x3F));
			ret.push_back(0x80 | ((code >> 6) & 0x3F));
			ret.push_back(0x80 | (code & 0x3F));
		}
		else
		{
			throw runtime_error{ "unicode error" };
		}
	}
	return ret;
}

void ChronoGramModel::buildSubwordTable()
{
	if (hp.subwordGrams == 0) return;
	unordered_map<string, uint32_t> tmpDict;
	vector<size_t> cnt;
	vector<pair<string, uint32_t>> orderedNgrams;

	vector<uint32_t> table;
	vector<size_t> ptrs;

	ptrs.emplace_back(0);
	for (size_t v = 0; v < vocabs.size(); ++v)
	{
		auto dw = decodeUTF8(vocabs.getStr(v));
		dw.insert(dw.begin(), 1);
		dw.push_back(2);
		if (dw.size() > hp.subwordGrams)
		{
			for (size_t i = hp.subwordGrams - 1; i < dw.size(); ++i)
			{
				auto ngram = encodeUTF8(dw.substr(i + 1 - hp.subwordGrams, hp.subwordGrams));
				auto it = tmpDict.find(ngram);
				if (it == tmpDict.end())
				{
					it = tmpDict.emplace(ngram, tmpDict.size()).first;
					cnt.emplace_back(0);
					orderedNgrams.emplace_back(ngram, orderedNgrams.size());
				}
				table.emplace_back(it->second);
				if(v < usedVocabSize()) cnt[it->second] += frequencies[v];
			}
		}
		ptrs.emplace_back(table.size());
	}

	sort(orderedNgrams.begin(), orderedNgrams.end(), [&](const pair<string, uint32_t>& a, const pair<string, uint32_t>& b)
	{
		return cnt[tmpDict.find(a.first)->second] > cnt[tmpDict.find(b.first)->second];
	});

	size_t subwordSize = find(cnt.begin(), cnt.end(), 0) - cnt.begin();

	vector<size_t> remapper(orderedNgrams.size());
	for (auto& p : orderedNgrams)
	{
		if(subwordVocabs.size() < subwordSize) subwordVocabs.add(p.first);
		remapper[p.second] = &p - &orderedNgrams[0];
	}

	for (auto& v : subwordTable)
	{
		v = remapper[v];
	}

	subwordTablePtrs.emplace_back(0);
	for (size_t i = 1; i < ptrs.size(); ++i)
	{
		for (size_t p = ptrs[i - 1]; p < ptrs[i]; ++p)
		{
			uint32_t np = remapper[table[p]];
			if (np < subwordSize
				&& find(subwordTable.begin() + subwordTablePtrs.back(), subwordTable.end(), np) == subwordTable.end()
				)
			{
				subwordTable.emplace_back(np);
			}
		}
		subwordTablePtrs.emplace_back(subwordTable.size());
	}

	subwordIn = Eigen::MatrixXf::Zero(hp.dimension, hp.order * subwordVocabs.size());
	if (hp.useSubwordOut)
	{
		subwordOut = Eigen::MatrixXf::Zero(hp.dimension, subwordVocabs.size());
	}
}

VectorXf ChronoGramModel::makeCoef(size_t r, float z)
{
	VectorXf coef = VectorXf::Zero(r);
	for (size_t i = 0; i < r; ++i)
	{
		coef[i] = poly::chebyshevTGet(i, 2 * z - 1);
	}
	return coef;
}

VectorXf ChronoGramModel::makeDCoef(size_t r, float z)
{
	VectorXf coef = VectorXf::Zero(r - 1);
	for (size_t i = 1; i < r; ++i)
	{
		coef[i - 1] = 2 * poly::chebyshevTDerived(i, 2 * z - 1);
	}
	return coef;
}

const VectorXf& ChronoGramModel::makeTimedVector(size_t wv, const VectorXf& coef) const
{
	thread_local VectorXf ret;
	if (hp.subwordGrams)
	{
		thread_local MatrixXf m;
		m = MatrixXf::Zero(hp.dimension, hp.order);
		for (size_t p = subwordTablePtrs[wv]; p < subwordTablePtrs[wv + 1]; ++p)
		{
			m += subwordIn.block(0, subwordTable[p] * hp.order, hp.dimension, hp.order);
		}
		if(wv < usedVocabSize()) m += in.block(0, wv * hp.order, hp.dimension, hp.order);
		return ret = (m * coef);
	}
	else
	{
		return ret = (in.block(0, wv * hp.order, hp.dimension, hp.order) * coef);
	}
}

const VectorXf& ChronoGramModel::makeTimedVectorDropout(size_t wv, const Eigen::VectorXf& coef, const std::vector<uint32_t>& subwords) const
{
	thread_local VectorXf ret;
	if (subwords.empty())
	{
		if (wv < usedVocabSize()) return ret = in.block(0, wv * hp.order, hp.dimension, hp.order) * coef;
		else return ret.setZero(), ret;
	}
	thread_local MatrixXf m;
	m = MatrixXf::Zero(hp.dimension, hp.order);
	for (auto p : subwords)
	{
		m += subwordIn.block(0, p * hp.order, hp.dimension, hp.order);
	}
	m *= (float)(subwordTablePtrs[wv + 1] - subwordTablePtrs[wv]) / subwords.size();
	if (wv < usedVocabSize()) m += in.block(0, wv * hp.order, hp.dimension, hp.order);
	return ret = m * coef;
}

const VectorXf& ChronoGramModel::makeSubwordTimedVector(const vector<uint32_t>& swv, const VectorXf& coef) const
{
	thread_local VectorXf ret;
	if (hp.subwordGrams)
	{
		thread_local MatrixXf m;
		m = MatrixXf::Zero(hp.dimension, hp.order);
		for (auto p : swv)
		{
			m += subwordIn.block(0, p * hp.order, hp.dimension, hp.order);
		}
		return ret = (m * coef);
	}
	else
	{
		return ret = VectorXf::Zero(hp.dimension);
	}
}

vector<uint32_t> ChronoGramModel::getSubwordIds(const std::string& s) const
{
	vector<uint32_t> ret;

	auto dw = decodeUTF8(s);
	dw.insert(dw.begin(), 1);
	dw.push_back(2);
	if (dw.size() <= hp.subwordGrams) return ret;
	for (size_t i = hp.subwordGrams - 1; i < dw.size(); ++i)
	{
		uint32_t swId = subwordVocabs.get(encodeUTF8(dw.substr(i + 1 - hp.subwordGrams, hp.subwordGrams)));
		if (swId == subwordVocabs.npos) continue;
		ret.emplace_back(swId);
	}

	return ret;
}

template<bool _initialization>
float ChronoGramModel::inplaceUpdate(size_t x, size_t y, float lr, bool negative, const VectorXf& lWeight, const vector<uint32_t>& subwords)
{
	thread_local VectorXf inSum, inGrad, dropout;
	auto outcol = out.col(y);
	inSum = _initialization ? in.col(x * hp.order) : makeTimedVectorDropout(x, lWeight, subwords);
	if (hp.dropout)
	{
		dropout = dropoutGen.generateLike(inSum, globalData.vrg);
		if (dropout.maxCoeff() == 0) dropout.setOnes();
		inSum.array() *= dropout.array() / (dropout.sum() / dropout.size());
	}
	float f = inSum.dot(outcol);
	float pr = logsigmoid(f * (negative ? -1 : 1));

	float d = ((negative ? 0 : 1) - sigmoid(f));
	float g = lr * d;

	inGrad = g * outcol;
	auto outGrad = g * inSum;
	outcol += outGrad;
	if (hp.dropout)
	{
		inGrad.array() *= dropout.array();
	}

	if (_initialization || fixedWords.count(x))
	{
		if (x < usedVocabSize()) in.col(x * hp.order) += inGrad;
	}
	else
	{
		if (subwords.empty())
		{
			if (x < usedVocabSize()) in.block(0, x * hp.order, hp.dimension, hp.order) += inGrad * lWeight.transpose();
		}
		else
		{
			thread_local MatrixXf inGradM;
			inGradM = inGrad * lWeight.transpose();
			if (x < usedVocabSize()) in.block(0, x * hp.order, hp.dimension, hp.order) += inGradM;
			for (auto p : subwords)
			{
				subwordIn.block(0, p * hp.order, hp.dimension, hp.order) += inGradM * ((float)(subwordTablePtrs[x + 1] - subwordTablePtrs[x]) / subwords.size());
			}
		}
	}
	return pr;
}

template<bool _initialization>
float ChronoGramModel::getUpdateGradient(size_t x, size_t y, float lr, bool negative, const VectorXf& lWeight, const vector<uint32_t>& subwords,
	Eigen::DenseBase<Eigen::MatrixXf>::ColXpr xGrad, Eigen::DenseBase<Eigen::MatrixXf>::ColXpr yGrad, ThreadLocalData& ld)
{
	thread_local VectorXf inSum, dropout;
	auto outcol = out.col(y);
	inSum = _initialization ? in.col(x * hp.order) : makeTimedVectorDropout(x, lWeight, subwords);
	if (hp.dropout)
	{
		dropout = dropoutGen.generateLike(inSum, ld.vrg);
		if (dropout.maxCoeff() == 0) dropout.setOnes();
		inSum.array() *= dropout.array() / (dropout.sum() / dropout.size());
	}
	float f = inSum.dot(outcol);
	float pr = logsigmoid(f * (negative ? -1 : 1));
	float d = ((negative ? 0 : 1) - sigmoid(f));
	float g = lr * d;

	if (hp.dropout)
	{
		xGrad += g * (outcol.array() * dropout.array()).matrix();
	}
	else
	{
		xGrad += g * outcol;
	}
	yGrad += g * inSum;
	
	return max(pr, -100.f);
}

float ChronoGramModel::inplaceTimeUpdate(size_t x, float lr, const VectorXf& lWeight, const vector<uint32_t>& subwords,
	const vector<VectorXf>& randSampleWeight)
{
	thread_local VectorXf atv, atv2, dropout;
	atv = makeTimedVectorDropout(x, lWeight, subwords).topRows(hp.ugPartDimension);
	if (hp.dropout)
	{
		dropout = dropoutGen.generateLike(atv, globalData.vrg);
		if (dropout.maxCoeff() == 0) dropout.setOnes();
		atv.array() *= dropout.array() / (dropout.sum() / dropout.size());
	}
	auto t = TempAct{}(atv.squaredNorm() / 2);
	float ll = log(t.first + 1e-12f);
	auto d = atv * lWeight.transpose() * (t.second / (t.first + 1e-12f));
	float s = 0;
	MatrixXf nd = MatrixXf::Zero(hp.dimension, hp.order);
	for (auto& r : randSampleWeight)
	{
		atv2 = makeTimedVectorDropout(x, r, subwords).topRows(hp.ugPartDimension);
		if (hp.dropout)
		{
			dropout = dropoutGen.generateLike(atv2, globalData.vrg);
			if (dropout.maxCoeff() == 0) dropout.setOnes();
			atv2.array() *= dropout.array() / (dropout.sum() / dropout.size());
		}
		auto t2 = TempAct{}(atv2.squaredNorm() / 2);
		nd += atv2 * r.transpose() * t2.second;
		s += t2.first + 1e-12f;
	}
	ll -= log(s / randSampleWeight.size());
	if (subwords.empty())
	{
		if (x < usedVocabSize()) in.block(0, x * hp.order, hp.ugPartDimension, hp.order) += (d - nd / s) * lr;
	}
	else
	{
		thread_local MatrixXf g;
		g = (d - nd / s) * lr;
		if (x < usedVocabSize()) in.block(0, x * hp.order, hp.ugPartDimension, hp.order) += g;
		for (auto p : subwords)
		{
			subwordIn.block(0, p * hp.order, hp.ugPartDimension, hp.order) += g * ((float)(subwordTablePtrs[x + 1] - subwordTablePtrs[x]) / subwords.size());
		}
	}
	return ll;
}

float ChronoGramModel::getTimeUpdateGradient(size_t x, float lr, const VectorXf & lWeight, const vector<uint32_t>& subwords,
	const vector<VectorXf>& randSampleWeight, Block<MatrixXf> grad)
{
	thread_local VectorXf atv, atv2, dropout;
	atv = makeTimedVectorDropout(x, lWeight, subwords).topRows(hp.ugPartDimension);
	if (hp.dropout)
	{
		dropout = dropoutGen.generateLike(atv, globalData.vrg);
		if (dropout.maxCoeff() == 0) dropout.setOnes();
		atv.array() *= dropout.array() / (dropout.sum() / dropout.size());
	}
	auto t = TempAct{}(atv.squaredNorm() / 2);
	float ll = log(t.first + 1e-12f);
	auto d = atv * lWeight.transpose() * (t.second / (t.first + 1e-12));
	float s = 0;
	MatrixXf nd = MatrixXf::Zero(hp.ugPartDimension, hp.order);
	for (auto& r : randSampleWeight)
	{
		atv2 = makeTimedVectorDropout(x, r, subwords).topRows(hp.ugPartDimension);
		if (hp.dropout)
		{
			dropout = dropoutGen.generateLike(atv2, globalData.vrg);
			if (dropout.maxCoeff() == 0) dropout.setOnes();
			atv2.array() *= dropout.array() / (dropout.sum() / dropout.size());
		}
		auto t2 = TempAct{}(atv2.squaredNorm() / 2);
		nd += atv2 * r.transpose() * t2.second;
		s += t2.first + 1e-12f;
	}
	ll -= log(s / randSampleWeight.size());
	grad.topRows(hp.ugPartDimension) += (d - nd / s) * lr;
	return ll;
}

float ChronoGramModel::updateTimePrior(float lr, const VectorXf & lWeight, const vector<const VectorXf*>& randSampleWeight)
{
	thread_local VectorXf nd;
	float p = timePrior.dot(lWeight);
	auto t = TempAct{}(p * p / 2);
	float ll = log(t.first + 1e-12f);
	auto d = lWeight * p;
	float s = 0;
	nd = VectorXf::Zero(hp.order);
	for (auto& r : randSampleWeight)
	{
		float dot = timePrior.dot(*r);
		auto t2 = TempAct{}(dot * dot / 2);
		nd += *r * dot * t2.second;
		s += t2.first + 1e-12f;
	}
	ll -= log(s / randSampleWeight.size());
	timePrior += (d * (t.second / t.first) - nd / s) * lr;
	return ll;
}

template<bool _Initialization, bool _GNgramMode>
ChronoGramModel::TrainResult ChronoGramModel::trainVectors(const uint32_t * ws, size_t N, float timePoint,
	size_t windowLen, float lr)
{
	TrainResult tr;
	tr.numWords = N;
	thread_local VectorXf coef;
	coef = makeCoef(hp.order, normalizedTimePoint(timePoint));
	for (size_t i = 0; i < N; ++i)
	{
		const auto& x = ws[i];
		if (_GNgramMode && i != 0 && i != N - 1) continue;
		if (_GNgramMode && x == (uint32_t)-1) continue;

		size_t jBegin = 0, jEnd = N - 1;
		if (i > windowLen) jBegin = i - windowLen;
		if (i + windowLen < N) jEnd = i + windowLen;

		thread_local vector<uint32_t> subwords;
		if (hp.subwordGrams)
		{
			subwords.clear();
			subwords.insert(subwords.end(), subwordTable.begin() + subwordTablePtrs[x], subwordTable.begin() + subwordTablePtrs[x + 1]);
			if (subwordTablePtrs[x + 1] - subwordTablePtrs[x] > hp.subwordDropoutRemain)
			{
				shuffle(subwords.begin(), subwords.end(), globalData.rg);
				subwords.erase(subwords.begin() + hp.subwordDropoutRemain, subwords.end());
			}
		}

		if (x < usedVocabSize() && inDecay[(size_t)x * hp.order] < 1)
		{
			in.block(0, (size_t)x * hp.order, hp.dimension, hp.order).array().rowwise() *= inDecay.segment((size_t)x * hp.order, hp.order).transpose();
			inDecay.segment((size_t)x * hp.order, hp.order) = 1;
		}

		for (auto p : subwords)
		{
			if (subwordInDecay[(size_t)p * hp.order] < 1)
			{
				subwordIn.block(0, (size_t)p * hp.order, hp.dimension, hp.order).array().rowwise() *= subwordInDecay.segment((size_t)p * hp.order, hp.order).transpose();
				subwordInDecay.segment((size_t)p * hp.order, hp.order) = 1;
			}
		}

		if (x < usedVocabSize() && outDecay[x] < 1)
		{
			out.col(x) *= outDecay[x];
			outDecay[x] = 1;
		}

		// update in, out vector
		for (auto j = jBegin; j <= jEnd; ++j)
		{
			if (i == j) continue;
			if (_GNgramMode && ws[j] == (uint32_t)-1) continue;
			if (ws[j] >= usedVocabSize()) continue;
			float ll = inplaceUpdate<_Initialization>(x, ws[j], lr, false, coef, subwords), llt = 0;
			assert(isfinite(ll));
			if (!_Initialization && hp.tnsWeight > 0)
			{
				for (size_t k = 0; k < hp.temporalNegativeSamples; ++k)
				{
					llt += inplaceUpdate<_Initialization>(x, ws[j], lr * hp.tnsWeight, true,
						makeCoef(hp.order, generate_canonical<float, 24>(globalData.rg)),
						subwords
					);
					assert(isfinite(llt));
				}
			}
			for (size_t k = 0; k < hp.negativeSamples; ++k)
			{
				uint32_t ns = unigramTable(globalData.rg);
				while (ns == ws[j]) ns = unigramTable(globalData.rg);
				ll += inplaceUpdate<_Initialization>(x, ns, lr, true, coef, subwords);
				assert(isfinite(ll));
			}
			tr.numPairs++;
			tr.contextLL += ll;
			tr.temporalLL += llt * hp.tnsWeight;
		}

		if (!_Initialization && hp.ugWeight > 0 && !fixedWords.count(x))
		{
			vector<VectorXf> randSampleWeight;
			for (size_t i = 0; i < hp.temporalNegativeSamples; ++i)
			{
				randSampleWeight.emplace_back(makeCoef(hp.order, generate_canonical<float, 24>(globalData.rg)));
			}
			tr.ugLL += inplaceTimeUpdate(x, lr * hp.ugWeight, coef, subwords, randSampleWeight) * hp.ugWeight;
		}
	}

	return tr;
}

template<bool _Initialization, bool _GNgramMode>
ChronoGramModel::TrainResult ChronoGramModel::trainVectorsMulti(const uint32_t * ws, size_t N, float timePoint,
	size_t windowLen, float lr, ThreadLocalData& ld, std::mutex* mtxIn, std::mutex* mtxSubwordIn, std::mutex* mtxOut, size_t hashSize)
{
	thread_local VectorXf coef;
	coef = makeCoef(hp.order, normalizedTimePoint(timePoint));
	TrainResult tr;
	tr.numWords = N;

	ld.updateOutIdx.clear();
	ld.updateOutIdxHash.clear();
	ld.updateOutMat = MatrixXf::Zero(hp.dimension, (hp.negativeSamples + 1) * windowLen * 2);
	thread_local MatrixXf updateIn, updateInBlock, updateUgIn;
	thread_local vector<float> updateSubwordUgCoef;
	float updateUgCoef = 0;
	updateIn = MatrixXf::Zero(hp.dimension, 1);
	updateInBlock = MatrixXf::Zero(hp.dimension, hp.order);
	for (size_t i = 0; i < N; ++i)
	{
		const auto x = ws[i];
		if (_GNgramMode && i != 0 && i != N - 1) continue;
		if (_GNgramMode && x == (uint32_t)-1) continue;

		size_t jBegin = 0, jEnd = N - 1;
		if (i > windowLen) jBegin = i - windowLen;
		if (i + windowLen < N) jEnd = i + windowLen;
		updateIn.setZero();

		thread_local vector<uint32_t> subwords;
		if (hp.subwordGrams)
		{
			subwords.clear();
			subwords.insert(subwords.end(), subwordTable.begin() + subwordTablePtrs[x], subwordTable.begin() + subwordTablePtrs[(size_t)x + 1]);
			if (subwordTablePtrs[x + 1] - subwordTablePtrs[x] > hp.subwordDropoutRemain)
			{
				shuffle(subwords.begin(), subwords.end(), ld.rg);
				subwords.erase(subwords.begin() + hp.subwordDropoutRemain, subwords.end());
			}
			if (x >= usedVocabSize() && subwords.empty()) continue;
		}

		lock_guard<mutex> lock(mtxIn[x % hashSize]);
		if (x < usedVocabSize() && inDecay[(size_t)x * hp.order] < 1)
		{
			in.block(0, (size_t)x * hp.order, hp.dimension, hp.order).array().rowwise() *= inDecay.segment((size_t)x * hp.order, hp.order).transpose();
			inDecay.segment((size_t)x * hp.order, hp.order) = 1;
		}

		// update in, out vector
		for (auto j = jBegin; j <= jEnd; ++j)
		{
			if (i == j) continue;
			if (_GNgramMode && ws[j] == (uint32_t)-1) continue;
			if (ws[j] >= usedVocabSize()) continue;
			if (ld.updateOutIdx.find(ws[j]) == ld.updateOutIdx.end())
			{
				ld.updateOutIdx.emplace(ws[j], ld.updateOutIdx.size());
				ld.updateOutIdxHash.emplace(ws[j] % hashSize);
			}

			float ll, llt = 0;
			{
				lock_guard<mutex> lock(mtxOut[ws[j] % hashSize]);
				ll = getUpdateGradient<_Initialization>(x, ws[j], 
					lr, false, 
					coef, subwords,
					updateIn.col(0),
					ld.updateOutMat.col(ld.updateOutIdx[ws[j]]),
					ld
				);
				assert(isfinite(ll));

				if (!_Initialization && hp.tnsWeight > 0)
				{
					for (size_t k = 0; k < hp.temporalNegativeSamples; ++k)
					{
						llt += getUpdateGradient<_Initialization>(x, ws[j],
							lr * hp.tnsWeight, true,
							makeCoef(hp.order, generate_canonical<float, 24>(ld.rg)),
							subwords,
							updateIn.col(0),
							ld.updateOutMat.col(ld.updateOutIdx[ws[j]]),
							ld
						);
						assert(isfinite(llt));
					}
				}
			}

			for (size_t k = 0; k < hp.negativeSamples; ++k)
			{
				uint32_t ns = unigramTable(ld.rg);
				while (ns == ws[j]) ns = unigramTable(ld.rg);
				if (ld.updateOutIdx.find(ns) == ld.updateOutIdx.end())
				{
					ld.updateOutIdx.emplace(ns, ld.updateOutIdx.size());
					ld.updateOutIdxHash.emplace(ns % hashSize);
				}
					
				lock_guard<mutex> lock(mtxOut[ns % hashSize]);
				ll += getUpdateGradient<_Initialization>(x, ns, 
					lr, 
					true, coef, subwords,
					updateIn.col(0),
					ld.updateOutMat.col(ld.updateOutIdx[ns]),
					ld
				);
				assert(isfinite(ll));
			}
			tr.numPairs++;
			tr.contextLL += ll;
			tr.temporalLL += llt * hp.tnsWeight;
		}

		if((!_Initialization && hp.ugWeight > 0 && !fixedWords.count(x)) || hp.subwordGrams) updateInBlock.setZero();
		if (!_Initialization && hp.ugWeight > 0 && !fixedWords.count(x))
		{
			thread_local vector<VectorXf> randSampleWeight;

			randSampleWeight.clear();
			for (size_t i = 0; i < hp.temporalNegativeSamples; ++i)
			{
				randSampleWeight.emplace_back(makeCoef(hp.order, generate_canonical<float, 24>(ld.rg)));
			}

			tr.ugLL += getTimeUpdateGradient(x, lr * hp.ugWeight, coef, subwords,
				randSampleWeight,
				updateInBlock.block(0, 0, hp.dimension, hp.order)
			) * hp.ugWeight;
		}

		{
			// deferred update of in-vector
			if (_Initialization || fixedWords.count(x))
			{
				if(x < usedVocabSize()) in.col((size_t)x * hp.order) += updateIn.col(0);
			}
			else if (hp.subwordGrams)
			{
				updateInBlock += updateIn * coef.transpose();
				if (x < usedVocabSize())
				{
					in.block(0, (size_t)x * hp.order, hp.dimension, hp.order) += updateInBlock;
				}
			}
			else if(x < usedVocabSize())
			{
				if (hp.ugWeight > 0) in.block(0, (size_t)x * hp.order, hp.dimension, hp.order) += updateInBlock;
				in.block(0, (size_t)x * hp.order, hp.dimension, hp.order) += updateIn * coef.transpose();
			}
		}

		if (hp.subwordGrams)
		{
			size_t si = 0;
			for (auto p : subwords)
			{
				lock_guard<mutex> lock(mtxSubwordIn[p % hashSize]);
				if (subwordInDecay[(size_t)p * hp.order] < 1)
				{
					subwordIn.block(0, (size_t)p * hp.order, hp.dimension, hp.order).array().rowwise() *= subwordInDecay.segment((size_t)p * hp.order, hp.order).transpose();
					subwordInDecay.segment((size_t)p * hp.order, hp.order) = 1;
				}
				subwordIn.block(0, (size_t)p * hp.order, hp.dimension, hp.order) += updateInBlock * ((float)(subwordTablePtrs[x + 1] - subwordTablePtrs[x]) / subwords.size());
			}
		}

		// deferred update of out-vector
		for(size_t hash : ld.updateOutIdxHash)
		{
			lock_guard<mutex> lock(mtxOut[hash]);
			for (auto& p : ld.updateOutIdx)
			{
				if (p.first % hashSize != hash) continue;
				if (outDecay[p.first] < 1)
				{
					out.col(p.first) *= outDecay[p.first];
					outDecay[p.first] = 1;
				}
				out.col(p.first) += ld.updateOutMat.col(p.second);
			}
		}
		ld.updateOutMat.setZero();
		ld.updateOutIdx.clear();
		ld.updateOutIdxHash.clear();
	}

	return tr;
}

void ChronoGramModel::trainTimePrior(const float * ts, size_t N, float lr, size_t report)
{
	unordered_map<float, VectorXf> coefMap;
	vector<float> randSample(hp.temporalNegativeSamples);
	vector<const VectorXf*> randSampleWeight(hp.temporalNegativeSamples);
	for (size_t i = 0; i < N; ++i)
	{
		for (size_t r = 0; r < hp.temporalNegativeSamples; ++r)
		{
			randSample[r] = generate_canonical<float, 24>(globalData.rg);
			if (!coefMap.count(randSample[r])) coefMap.emplace(randSample[r], makeCoef(hp.order, randSample[r]));
		}
		//float c_lr = max(lr * (1 - procTimePoints / (totalTimePoints + 1.f)), lr * 1e-4f) * 0.1f;
		auto it = coefMap.find(ts[i]);
		if (it == coefMap.end()) it = coefMap.emplace(ts[i], makeCoef(hp.order, normalizedTimePoint(ts[i]))).first;
		for (size_t r = 0; r < hp.temporalNegativeSamples; ++r)
		{
			randSampleWeight[r] = &coefMap.find(randSample[r])->second;
		}
		
		float ll = updateTimePrior(lr, it->second, randSampleWeight);
		procTimePoints++;
		timeLLCnt++;
		timeLL += (ll - timeLL) / timeLLCnt;
		if (report && procTimePoints % report == 0)
		{
			fprintf(stderr, "timePrior LL: %.4f\n", timeLL);
			/*for (size_t r = 0; r < hp.order; ++r)
			{
				fprintf(stderr, "%.4f ", timePrior[r]);
			}
			fprintf(stderr, "\n");*/
		}
	}
}

void ChronoGramModel::normalizeWordDist(bool updateVocab, ThreadPool* pool)
{
	constexpr size_t step = 128;

	vector<VectorXf> coefs;
	for (size_t i = 0; i <= step; ++i)
	{
		coefs.emplace_back(makeCoef(hp.order, i * (1.f - timePadding * 2) / step + timePadding));
	}

	float p = 0;
	for (size_t i = 0; i <= step; ++i)
	{
		p += TempAct{}(pow(timePrior.dot(coefs[i]), 2) / 2).first;
	}
	p /= step + 1;
	timePriorScale = max(p, 1e-5f);
	if (updateVocab)
	{
		if (!pool || pool->getNumWorkers() == 1)
		{
			for (size_t v = 0; v < usedVocabSize(); ++v)
			{
				float p = 0;
				for (size_t i = 0; i <= step; ++i)
				{
					p += TempAct{}(makeTimedVector(v, coefs[i]).topRows(hp.ugPartDimension).squaredNorm() / 2).first;
				}
				p /= step + 1;
				wordScale[v] = p;
			}
		}
		else
		{
			const size_t chStride = pool->getNumWorkers();
			vector<future<void>> futures;
			for (size_t c = 0; c < chStride; ++c)
			{
				futures.emplace_back(pool->enqueue([&](size_t, size_t c)
				{
					for (size_t v = c; v < usedVocabSize(); v += chStride)
					{
						float p = 0;
						for (size_t i = 0; i <= step; ++i)
						{
							p += TempAct{}(makeTimedVector(v, coefs[i]).topRows(hp.ugPartDimension).squaredNorm() / 2).first;
						}
						p /= step + 1;
						wordScale[v] = p;
					}
				}, c));
			}
			for (auto& f : futures) f.get();
		}
	}
}

float ChronoGramModel::getTimePrior(const Eigen::VectorXf & coef) const
{
	return TempAct{}(pow(timePrior.dot(coef), 2) / 2).first / timePriorScale;
}

float ChronoGramModel::getWordProbByTime(uint32_t w, const Eigen::VectorXf & timedVector, const Eigen::VectorXf & coef, float tPrior) const
{
	return TempAct{}(timedVector.squaredNorm() / 2).first / wordScale[w];
}

float ChronoGramModel::getWordProbByTime(uint32_t w, float timePoint) const
{
	auto coef = makeCoef(hp.order, normalizedTimePoint(timePoint));
	thread_local MatrixXf m;
	m = MatrixXf::Zero(hp.ugPartDimension, hp.order);
	if (hp.subwordGrams)
	{
		for (size_t p = subwordTablePtrs[w]; p < subwordTablePtrs[w + 1]; ++p)
		{
			m += subwordIn.block(0, subwordTable[p] * hp.order, hp.ugPartDimension, hp.order);
		}
	}
	if (w < usedVocabSize()) m += in.block(0, w * hp.order, hp.ugPartDimension, hp.order);
	return TempAct{}((m * coef).squaredNorm() / 2).first / wordScale[w];
}


void ChronoGramModel::buildVocab(const std::function<ResultReader()>& reader, size_t minCnt, size_t minCntForSubword, size_t numWorkers)
{
	if (!numWorkers) numWorkers = thread::hardware_concurrency();

	if (!hp.subwordGrams) minCntForSubword = minCnt;

	float minT = INFINITY, maxT = -INFINITY;
	unordered_map<string, size_t> counter;
	if (numWorkers > 1)
	{
		vector<unordered_map<string, size_t>> counters(numWorkers - 1);
		{
			ThreadPool workers(numWorkers - 1, (numWorkers - 1) * 16);
			vector<string> words;
			auto riter = reader();
			while (1)
			{
				auto res = riter();
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
		auto riter = reader();
		while(1)
		{
			auto res = riter();
			if (res.stop) break;
			if (res.words.empty()) continue;
			minT = min(res.timePoint, minT);
			maxT = max(res.timePoint, maxT);
			for (auto& w : res.words) ++counter[w];
		}
	}
	zBias = minT;
	zSlope = minT == maxT ? 1 : 1 / (maxT - minT);

	vector<pair<size_t, string>> ordered;
	for (auto& p : counter) ordered.emplace_back(p.second, p.first);
	sort(ordered.rbegin(), ordered.rend());

	for (auto& p : ordered)
	{
		if (p.first < minCntForSubword) continue;
		if (p.first >= minCnt)
		{
			frequencies.emplace_back(p.first);
		}
		vocabs.add(p.second);
	}
	buildModel();
}

size_t ChronoGramModel::recountVocab(const std::function<ResultReader()>& reader, float minT, float maxT, size_t numWorkers)
{
	if (!numWorkers) numWorkers = thread::hardware_concurrency();
	vector<vector<uint64_t>> counters(numWorkers);
	for (auto& c : counters)
	{
		c.resize(usedVocabSize());
	}

	{
		ThreadPool workers(numWorkers, numWorkers * 16);
		vector<string> words;
		auto riter = reader();
		while (1)
		{
			auto res = riter();
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
		for (size_t n = 0; n < usedVocabSize(); ++n) frequencies[n] += counters[i][n];
	}
	buildTable();
	return count_if(frequencies.begin(), frequencies.end(), [](size_t e) { return e; });
}


size_t ChronoGramModel::recountVocab(const std::function<GNgramResultReader()>& reader, float minT, float maxT, size_t numWorkers)
{
	if (!numWorkers) numWorkers = thread::hardware_concurrency();
	fill(frequencies.begin(), frequencies.end(), 0);
	auto riter = reader();
	while (1)
	{
		auto res = riter();
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
	return count_if(frequencies.begin(), frequencies.end(), [](size_t e) { return e; });
}


bool ChronoGramModel::addFixedWord(const std::string & word)
{
	size_t wv = vocabs.get(word);
	if (wv == (size_t)-1) return false;
	fixedWords.emplace(wv);
	in.block(0, wv * hp.order + 1, hp.dimension, hp.order - 1).setZero();
	return true;
}

void ChronoGramModel::buildVocabFromDict(const function<pair<string, uint64_t>()>& reader, float minT, float maxT, size_t vocabSize)
{
	if (!hp.subwordGrams) vocabSize = -1;
	zBias = minT;
	zSlope = minT == maxT ? 1 : 1 / (maxT - minT);
	pair<string, uint64_t> p;
	while ((p = reader()).second > 0)
	{
		if(vocabs.size() < vocabSize) frequencies.emplace_back(p.second);
		vocabs.add(p.first);
	}
	buildModel();
}

bool ChronoGramModel::defaultReportCallback(size_t steps, float progress, float contextLL, float temporalLL, float ugLL, float lr, float timePerKword)
{
	fprintf(stderr, "%.2f%% %.4f %.4f %.4f %.4f %.4f %.2f kwords/sec\n",
		progress * 100, contextLL + temporalLL + ugLL, contextLL, temporalLL, ugLL,
		lr,
		timePerKword);
	fflush(stderr);
	return false;
}

template<bool _multi>
void ChronoGramModel::updateWeightDecay(float lr, mutex* mtxIn, mutex* mtxSubwordIn, mutex* mtxOut, size_t hashSize)
{
	if (hp.weightDecay)
	{
		for (size_t h = 0; h < hashSize; ++h)
		{
			{
				optional_lock<_multi> lg(mtxIn[h]);
				for (size_t i = h; i < usedVocabSize(); i += hashSize)
				{
					inDecay.segment(i * hp.order, hp.order) *= Eigen::ArrayXf::LinSpaced(hp.order, 1 - lr * hp.weightDecay, 1 - lr * hp.weightDecay * hp.order);
				}
			}
			{
				optional_lock<_multi> lg(mtxOut[h]);
				for (size_t i = h; i < usedVocabSize(); i += hashSize)
				{
					outDecay[i] *= 1 - lr * hp.weightDecay;
				}
			}
		}
	}

	if (hp.subwordGrams && hp.subwordWeightDecay)
	{
		for (size_t h = 0; h < hashSize; ++h)
		{
			optional_lock<_multi> lg(mtxSubwordIn[h]);
			for (size_t i = h; i < subwordVocabs.size(); i += hashSize)
			{
				subwordInDecay.segment(i * hp.order, hp.order) *= 1 - lr * hp.subwordWeightDecay;
			}
		}
	}
}

template<bool _Initialization>
void ChronoGramModel::train(const function<ResultReader()>& reader,
	size_t numWorkers, size_t windowLen, float start_lr, float end_lr, size_t batch,
	float epochs, size_t report, const ReportCallback& reportCallback)
{
	if (!numWorkers) numWorkers = thread::hardware_concurrency();
	if (!pool || pool->getNumWorkers() != numWorkers)
	{
		pool = unique_ptr<ThreadPool>(new ThreadPool(numWorkers));
	}

	const size_t hashMul = 64;
	vector<ThreadLocalData> ld;
	unique_ptr<mutex[]> mtxIn, mtxSubwordIn, mtxOut;
	if (numWorkers > 1)
	{
		ld.resize(numWorkers);
		for (auto& l : ld)
		{
			l.rg = mt19937_64{ globalData.rg() };
			l.vrg = Eigen::Rand::Vmt19937_64{ globalData.rg() };
		}
		mtxIn = unique_ptr<mutex[]>(new mutex[numWorkers * hashMul]);
		mtxOut = unique_ptr<mutex[]>(new mutex[numWorkers * hashMul]);
		if(hp.subwordGrams) mtxSubwordIn = unique_ptr<mutex[]>(new mutex[numWorkers * hashMul]);
	}
	vector<pair<vector<uint32_t>, float>> collections;
	vector<float> timePoints;
	const float minT = getMinPoint(), maxT = getMaxPoint();
	uint64_t totW = accumulate(frequencies.begin(), frequencies.end(), 0ull);
	size_t updatedWords = 0;

	inDecay = Eigen::ArrayXf::Ones(in.cols());
	outDecay = Eigen::ArrayXf::Ones(out.cols());
	subwordInDecay = Eigen::ArrayXf::Ones(subwordIn.cols());

	// estimate total size
	totalWords = totW * (double)epochs;
	totalTimePoints = totalWords / 4;
	procWords = lastProcWords = 0;
	size_t totalLLCnt = 0;
	double contextLL = 0, temporalLL = 0, ugLL = 0;
	timeLL = timeLLCnt = 0;
	procTimePoints = 0;
	timer.reset();

	const auto& procCollection = [&]()
	{
		if (collections.empty()) return false;
		shuffle(collections.begin(), collections.end(), globalData.rg);
		for (auto& c : collections)
		{
			size_t tpCnt = (c.first.size() + 2) / 4;
			if(tpCnt) timePoints.resize(timePoints.size() + tpCnt, c.second);
		}
		if(!_Initialization) shuffle(timePoints.begin(), timePoints.end(), globalData.rg);

		float lr = start_lr + (end_lr - start_lr) * min(procWords / (float)totalWords, 1.f);
		constexpr float timeLR = 0.1f;
		if (numWorkers > 1)
		{
			vector<future<TrainResult>> futures;
			futures.reserve(collections.size());
			for (auto& d : collections)
			{
				if (hp.temporalSubsampling > 0)
				{
					float p = hp.temporalSubsampling / getTimePrior(d.second);
					if (p < 1 && generate_canonical<float, 24>(globalData.rg) > sqrt(p) + p)
					{
						procWords += d.first.size();
						continue;
					}
				}
				futures.emplace_back(pool->enqueue([&](size_t threadId)
				{
					return trainVectorsMulti<_Initialization>(d.first.data(), d.first.size(), d.second,
						windowLen, lr, ld[threadId], mtxIn.get(), mtxSubwordIn.get(), mtxOut.get(), numWorkers * hashMul);
				}));
			}
			if(!_Initialization) trainTimePrior(timePoints.data(), timePoints.size(), lr * timeLR, report);
			for (auto& f : futures)
			{
				TrainResult tr = f.get();
				totalLLCnt += tr.numPairs;
				contextLL += (tr.contextLL - tr.numPairs * contextLL) / totalLLCnt;
				temporalLL += (tr.temporalLL - tr.numPairs * temporalLL) / totalLLCnt;
				ugLL += (tr.ugLL - tr.numPairs * ugLL) / totalLLCnt;
				procWords += tr.numWords;

				if (hp.weightDecayInterval
					&& updatedWords / hp.weightDecayInterval < (updatedWords + tr.numWords) / hp.weightDecayInterval)
				{
					updateWeightDecay<true>(lr, mtxIn.get(), mtxSubwordIn.get(), mtxOut.get(), numWorkers * hashMul);
				}
				updatedWords += tr.numWords;
			}
			
			bool result = false;
			if (report && lastProcWords / report < procWords / report)
			{
				normalizeWordDist(true, pool.get());
				float timePerKword = (procWords - lastProcWords) / timer.getElapsed() / 1000.f;
				if (reportCallback)
				{
					result = reportCallback(procWords, procWords / (double)totalWords, contextLL, temporalLL, ugLL, lr, timePerKword);
				}
				lastProcWords = procWords;
				timer.reset();
			}
			if(result) return result;
			if (!_Initialization)
			{
				normalizeWordDist(false);
			}
		}
		else
		{
			for (auto& d : collections)
			{
				if (hp.temporalSubsampling > 0)
				{
					float p = hp.temporalSubsampling / getTimePrior(d.second);
					if (p < 1 && generate_canonical<float, 24>(globalData.rg) > sqrt(p) + p)
					{
						procWords += d.first.size();
						continue;
					}
				}
				TrainResult tr = trainVectors<_Initialization>(d.first.data(), d.first.size(), d.second,
					windowLen, lr);
				totalLLCnt += tr.numPairs;
				contextLL += (tr.contextLL - tr.numPairs * contextLL) / totalLLCnt;
				temporalLL += (tr.temporalLL - tr.numPairs * temporalLL) / totalLLCnt;
				ugLL += (tr.ugLL - tr.numPairs * ugLL) / totalLLCnt;
				procWords += tr.numWords;

				if (hp.weightDecayInterval
					&& updatedWords / hp.weightDecayInterval < (updatedWords + tr.numWords) / hp.weightDecayInterval)
				{
					updateWeightDecay<false>(lr, mtxIn.get(), mtxSubwordIn.get(), mtxOut.get(), numWorkers * hashMul);
				}
				updatedWords += tr.numWords;

				bool result = false;
				if (report && lastProcWords / report < procWords / report)
				{
					normalizeWordDist();
					float timePerKword = (procWords - lastProcWords) / timer.getElapsed() / 1000.f;
					if (reportCallback)
					{
						result = reportCallback(procWords, procWords / (double)totalWords, contextLL, temporalLL, ugLL, lr, timePerKword);
					}
					lastProcWords = procWords;
					timer.reset();
				}
				if (result) return result;
			}
			if (!_Initialization)
			{
				trainTimePrior(timePoints.data(), timePoints.size(), lr * timeLR, report);
				normalizeWordDist(false);
			}
		}
		collections.clear();
		timePoints.clear();
		return false;
	};

	size_t flr = epochs;
	float frac = fmod(epochs, 1.f);
	for (size_t e = 0; e <= flr; ++e)
	{
		auto riter = reader();
		while(1)
		{
			if (e == flr && procWords >= frac * totW) break;
			auto rresult = riter();
			if (rresult.words.empty()) break;
			if (rresult.timePoint < minT || rresult.timePoint > maxT) continue;

			vector<uint32_t> doc;
			doc.reserve(rresult.words.size());
			for (auto& w : rresult.words)
			{
				auto id = vocabs.get(w);
				if (id < 0) continue;
				float ww = id < usedVocabSize() ? (hp.subsampling / (frequencies[id] / (float)totW)) : 1;
				if (hp.subsampling > 0 &&
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

	if(!_Initialization) normalizeWordDist(true, pool.get());
}

template<bool _Initialization>
void ChronoGramModel::trainFromGNgram(const function<GNgramResultReader()>& reader, uint64_t maxItems,
	size_t numWorkers, float start_lr, float end_lr, size_t batchSents, float epochs, size_t report,
	const ReportCallback& reportCallback)
{
	if (!numWorkers) numWorkers = thread::hardware_concurrency();
	if (!pool || pool->getNumWorkers() != numWorkers)
	{
		pool = unique_ptr<ThreadPool>(new ThreadPool(numWorkers));
	}
	const size_t hashMul = 64;
	vector<ThreadLocalData> ld;
	unique_ptr<mutex[]> mtxIn, mtxSubwordIn, mtxOut;
	if (numWorkers > 1)
	{
		ld.resize(numWorkers);
		for (auto& l : ld)
		{
			l.rg = mt19937_64{ globalData.rg() };
		}
		mtxIn = unique_ptr<mutex[]>(new mutex[numWorkers * hashMul]);
		mtxOut = unique_ptr<mutex[]>(new mutex[numWorkers * hashMul]);
		if (hp.subwordGrams) mtxSubwordIn = unique_ptr<mutex[]>(new mutex[numWorkers * hashMul]);
	}

	vector<pair<array<uint32_t, 5>, float>> collections;
	vector<float> timePoints;
	size_t updatedWords = 0, lastUpdatedWords = 0;
	
	const float minT = getMinPoint(), maxT = getMaxPoint();
	uint64_t totW = accumulate(frequencies.begin(), frequencies.end(), 0ull);

	procWords = lastProcWords = 0;
	totalWords = maxItems * (double)epochs;
	totalTimePoints = totalWords / 8;
	size_t totalLLCnt = 0;
	timeLL = timeLLCnt = 0;
	double contextLL = 0, temporalLL = 0, ugLL = 0;
	procTimePoints = 0;

	inDecay = Eigen::ArrayXf::Ones(in.cols());
	outDecay = Eigen::ArrayXf::Ones(out.cols());
	subwordInDecay = Eigen::ArrayXf::Ones(subwordIn.cols());

	timer.reset();

	const auto& procCollection = [&](bool last) -> bool
	{
		if (collections.empty()) return false;
		shuffle(collections.begin(), collections.end(), globalData.rg);

		if (!_Initialization)
		{
			timePoints.resize(min(batchSents, collections.size()) / 8);
			transform(collections.begin() + (last ? 0 : batchSents * 2), 
				collections.begin() + (last ? 0 : batchSents * 2) + timePoints.size(),
				timePoints.begin(), 
				[](const pair<array<uint32_t, 5>, float>& p) { return p.second; }
			);
		}

		float lr = start_lr + (end_lr - start_lr) * (procWords / (float)totalWords);
		constexpr float timeLR = 0.1f;
		if (numWorkers > 1)
		{
			vector<future<TrainResult>> futures;
			futures.reserve(batchSents);
			for (size_t i = last ? 0 : batchSents * 2; i < collections.size(); ++i)
			{
				auto& d = collections[i];
				if (hp.temporalSubsampling > 0)
				{
					float p = hp.temporalSubsampling / getTimePrior(d.second);
					if (p < 1 && generate_canonical<float, 24>(globalData.rg) > sqrt(p) + p)
					{
						continue;
					}
				}
				futures.emplace_back(pool->enqueue([&](size_t threadId)
				{
					return trainVectorsMulti<_Initialization, true>(d.first.data(), 5, d.second,
						4, lr, ld[threadId], mtxIn.get(), mtxSubwordIn.get(), mtxOut.get(), numWorkers * hashMul);
				}));
			}
			if (!_Initialization) trainTimePrior(timePoints.data(), timePoints.size(), lr * timeLR, report);
			for (auto& f : futures)
			{
				TrainResult tr = f.get();
				totalLLCnt += tr.numPairs;
				contextLL += (tr.contextLL - tr.numPairs * contextLL) / totalLLCnt;
				temporalLL += (tr.temporalLL - tr.numPairs * temporalLL) / totalLLCnt;
				ugLL += (tr.ugLL - tr.numPairs * ugLL) / totalLLCnt;

				if (hp.weightDecayInterval 
					&& updatedWords / hp.weightDecayInterval < (updatedWords + tr.numWords) / hp.weightDecayInterval)
				{
					updateWeightDecay<true>(lr, mtxIn.get(), mtxSubwordIn.get(), mtxOut.get(), numWorkers * hashMul);
				}
				updatedWords += tr.numWords;
			}
			if (!_Initialization)
			{
				normalizeWordDist(false);
			}
		}
		else
		{
			for (size_t i = last ? 0 : batchSents * 2; i < collections.size(); ++i)
			{
				auto& d = collections[i];
				if (hp.temporalSubsampling > 0)
				{
					float p = hp.temporalSubsampling / getTimePrior(d.second);
					if (p < 1 && generate_canonical<float, 24>(globalData.rg) > sqrt(p) + p)
					{
						continue;
					}
				}
				TrainResult tr = trainVectors<_Initialization, true>(d.first.data(), 5, d.second, 4, lr);
				totalLLCnt += tr.numPairs;
				contextLL += (tr.contextLL - tr.numPairs * contextLL) / totalLLCnt;
				temporalLL += (tr.temporalLL - tr.numPairs * temporalLL) / totalLLCnt;
				ugLL += (tr.ugLL - tr.numPairs * ugLL) / totalLLCnt;

				if (hp.weightDecayInterval
					&& updatedWords / hp.weightDecayInterval < (updatedWords + tr.numWords) / hp.weightDecayInterval)
				{
					updateWeightDecay<false>(lr, mtxIn.get(), mtxSubwordIn.get(), mtxOut.get(), numWorkers * hashMul);
				}
				updatedWords += tr.numWords;
			}
			if (!_Initialization)
			{
				trainTimePrior(timePoints.data(), timePoints.size(), lr * timeLR, report);
				normalizeWordDist(false);
			}
		}
		collections.erase(collections.begin() + (last ? 0 : batchSents * 2), collections.end());
		timePoints.clear();

		bool result = false;
		if (report && lastProcWords / report <  procWords / report)
		{
			for (size_t i = 0; i < subwordVocabs.size(); ++i)
			{
				if (subwordInDecay[i * hp.order] < 1)
				{
					subwordIn.block(0, i * hp.order, hp.dimension, hp.order).array().rowwise() *= subwordInDecay.segment(i * hp.order, hp.order).transpose();
					subwordInDecay.segment(i * hp.order, hp.order) = 1;
				}
			}

			normalizeWordDist(true, pool.get());
			float timePerKword = (procWords - lastProcWords) / timer.getElapsed() / 1000.f;
			if (reportCallback)
			{
				result = reportCallback(procWords, procWords / (double)totalWords, contextLL, temporalLL, ugLL, lr, timePerKword);
			}
			lastProcWords = procWords;
			timer.reset();
		}
		return result;
	};

	size_t flr = epochs;
	float frac = fmod(epochs, 1.f);
	for (size_t e = 0; e <= flr; ++e)
	{
		auto riter = reader();
		for (size_t id = 0; id < maxItems; ++id)
		{
			if (e == flr && id >= maxItems * frac) break;
			auto rresult = riter();
			if (rresult.yearCnt.empty()) break;

			for (auto& w : rresult.ngram)
			{
				if (w == (uint32_t)-1) continue;
				float ww = w < usedVocabSize() ? (hp.subsampling / (frequencies[w] / (float)totW)) : 1;
				if (hp.subsampling > 0 && ww < 1 &&
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
					collections.emplace_back(rresult.ngram, p.first);
				}
			}

			if (collections.size() >= batchSents * 3)
			{
				if (procCollection(false)) goto end_train;
			}
		}
	}
	procCollection(true);
end_train:
	for (size_t i = 0; i < usedVocabSize(); ++i)
	{
		if (inDecay[i * hp.order] < 1)
		{
			in.block(0, i * hp.order, hp.dimension, hp.order).array().rowwise() *= inDecay.segment(i * hp.order, hp.order).transpose();
			inDecay.segment(i * hp.order, hp.order) = 1;
		}
		if (outDecay[i] < 1)
		{
			out.col(i) *= outDecay[i];
			outDecay[i] = 1;
		}
	}
	for (size_t i = 0; i < subwordVocabs.size(); ++i)
	{
		if (subwordInDecay[i * hp.order] < 1)
		{
			subwordIn.block(0, i * hp.order, hp.dimension, hp.order).array().rowwise() *= subwordInDecay.segment(i * hp.order, hp.order).transpose();
			subwordInDecay.segment(i * hp.order, hp.order) = 1;
		}
	}

	if(!_Initialization) normalizeWordDist(true, pool.get());
}

float ChronoGramModel::arcLengthOfWord(const string & word, size_t step) const
{
	size_t wv = vocabs.get(word);
	if (wv == (size_t)-1) return {};
	float len = 0;
	VectorXf v = makeTimedVector(wv, makeCoef(hp.order, timePadding));
	for (size_t i = 0; i < step; ++i)
	{
		VectorXf u = makeTimedVector(wv, makeCoef(hp.order, (float)(i + 1) / step * (1 - timePadding * 2) + timePadding));
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
	VectorXf v = makeTimedVector(wv, makeCoef(hp.order, timePadding));
	for (size_t i = 0; i < step; ++i)
	{
		VectorXf u = makeTimedVector(wv, makeCoef(hp.order, (float)(i + 1) / step * (1 - timePadding * 2) + timePadding));
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
	float searchingTimePoint, float m, size_t K, bool normalize) const
{
	VectorXf vec = VectorXf::Zero(hp.dimension);
	const size_t V = usedVocabSize();
	unordered_set<size_t> uniqs;
	VectorXf coef = makeCoef(hp.order, normalizedTimePoint(searchingTimePoint));
	for (auto& p : positiveWords)
	{
		size_t wv = vocabs.get(p.first);
		if (wv == (size_t)-1 && !hp.subwordGrams) return {};
		VectorXf cf = makeCoef(hp.order, normalizedTimePoint(p.second));
		float tPrior = getTimePrior(cf);
		VectorXf tv;
		if (wv < V) tv = makeTimedVector(wv, cf);
		else tv = makeSubwordTimedVector(getSubwordIds(p.first), cf);
		//float wPrior = getWordProbByTime(wv, tv, cf, tPrior);
		//if (hp.zeta > 0 && wPrior / (tPrior + tpvBias) < tpvThreshold) return {};
		VectorXf v = tv * (1 - m);
		if (wv < V) v += out.col(wv) * m;
		if (normalize) v /= v.norm() + 1e-3f;
		vec += v;
		uniqs.emplace(wv);
	}

	for (auto& p : negativeWords)
	{
		size_t wv = vocabs.get(p.first);
		if (wv == (size_t)-1 && !hp.subwordGrams) return {};
		VectorXf cf = makeCoef(hp.order, normalizedTimePoint(p.second));
		float tPrior = getTimePrior(cf);
		VectorXf tv;
		if (wv < V) tv = makeTimedVector(wv, cf);
		else tv = makeSubwordTimedVector(getSubwordIds(p.first), cf);
		//float wPrior = getWordProbByTime(wv, tv, cf, tPrior);
		//if (hp.zeta > 0 && wPrior / (tPrior + tpvBias) < tpvThreshold) return {};
		VectorXf v = tv * (1 - m);
		if (wv < V) v += out.col(wv) * m;
		if (normalize) v /= v.norm() + 1e-3f;
		vec -= v;
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
		float wPrior = getWordProbByTime(v, tv, coef, tPrior);
		if (hp.ugWeight > 0 && wPrior / (tPrior + tpvBias) < tpvThreshold)
		{
			sim[v] = make_pair(-INFINITY, wPrior / tPrior);
			continue;
		}
		sim[v] = make_pair((tv * (1 - m) + out.col(v) * m).normalized().dot(vec), wPrior / tPrior);
	}

	for (size_t k = 0; k < K; ++k)
	{
		size_t idx = max_element(sim.begin(), sim.end()) - sim.begin();
		if (sim.data()[idx].first == -INFINITY) break;
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
	VectorXf vec = VectorXf::Zero(hp.dimension);
	const size_t V = usedVocabSize();
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
		if (sim.data()[idx] == -INFINITY) break;
		top.emplace_back(vocabs.getStr(idx), sim.data()[idx]);
		sim.data()[idx] = -INFINITY;
	}
	return top;
}

vector<pair<string, float>> ChronoGramModel::calcShift(size_t minCnt, float time1, float time2, float m) const
{
	VectorXf coef1 = makeCoef(hp.order, normalizedTimePoint(time1)),
		coef2 = makeCoef(hp.order, normalizedTimePoint(time2));
	vector<pair<string, float>> ret;
	const size_t V = usedVocabSize();
	float tPrior1 = getTimePrior(coef1);
	float tPrior2 = getTimePrior(coef2);

	for (size_t v = 0; v < V; ++v)
	{
		if (frequencies[v] < minCnt) continue;
		VectorXf v1 = makeTimedVector(v, coef1);
		VectorXf v2 = makeTimedVector(v, coef2);
		if (hp.ugWeight > 0 && getWordProbByTime(v, v1, coef1, tPrior1) / (tPrior1 + tpvBias) < tpvThreshold) continue;
		if (hp.ugWeight > 0 && getWordProbByTime(v, v2, coef2, tPrior2) / (tPrior2 + tpvBias) < tpvThreshold) continue;
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
	
	sort(ret.begin(), ret.end(), [](const pair<string, float>& p1, const pair<string, float>& p2)
	{
		return p1.second < p2.second;
	});
	return ret;
}

float ChronoGramModel::sumSimilarity(const string & src, const vector<string>& targets, float timePoint, float m) const
{
	const size_t V = usedVocabSize();
	unordered_set<size_t> uniqs;
	VectorXf coef = makeCoef(hp.order, normalizedTimePoint(timePoint));
	size_t wv = vocabs.get(src);
	if (wv == (size_t)-1) return -INFINITY;
	float tPrior = getTimePrior(coef);
	VectorXf tv = makeTimedVector(wv, coef);
	float wPrior = getWordProbByTime(wv, tv, coef, tPrior);
	if (hp.zeta > 0 && wPrior / (tPrior + tpvBias) < tpvThreshold) return -INFINITY;
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
		float wPrior = getWordProbByTime(v, tv, coef, tPrior);
		if (hp.ugWeight > 0 && wPrior / (tPrior + tpvBias) < tpvThreshold) continue;
		sum += (tv * (1 - m) + out.col(v) * m).normalized().dot(vec);
	}

	return sum;
}

float ChronoGramModel::similarity(const string & word1, float time1, const string & word2, float time2) const
{
	size_t wv1 = vocabs.get(word1), wv2 = vocabs.get(word2);
	if (wv1 == (size_t)-1 || wv2 == (size_t)-1) return 0;
	VectorXf c1 = makeCoef(hp.order, normalizedTimePoint(time1)), c2 = makeCoef(hp.order, normalizedTimePoint(time2));
	return makeTimedVector(wv1, c1).normalized().dot(makeTimedVector(wv2, c2).normalized());
}

float ChronoGramModel::similarityStatic(const string & word1, const string & word2) const
{
	size_t wv1 = vocabs.get(word1), wv2 = vocabs.get(word2);
	if (wv1 == (size_t)-1 || wv2 == (size_t)-1) return 0;
	return out.col(wv1).normalized().dot(out.col(wv2).normalized());
}

MatrixXf ChronoGramModel::getEmbeddingMatrix(const string & word) const
{
	MatrixXf ret = MatrixXf::Zero(hp.dimension, hp.order);

	size_t wv = vocabs.get(word);
	if (wv == (size_t)-1)
	{
		if (!hp.subwordGrams) return {};
		for (auto sw : getSubwordIds(word))
		{
			ret += subwordIn.block(0, sw * hp.order, hp.dimension, hp.order);
		}
		return ret;
	}
	ret += in.block(0, wv * hp.order, hp.dimension, hp.order);
	if (hp.subwordGrams)
	{
		for (size_t p = subwordTablePtrs[wv]; p < subwordTablePtrs[wv + 1]; ++p)
		{
			ret += subwordIn.block(0, subwordTable[p] * hp.order, hp.dimension, hp.order);
		}
	}
	return ret;
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
	if (wv == (size_t)-1)
	{
		if (!hp.subwordGrams) return {};
		return makeSubwordTimedVector(getSubwordIds(word), makeCoef(hp.order, normalizedTimePoint(timePoint)));
	}
	return makeTimedVector(wv, makeCoef(hp.order, normalizedTimePoint(timePoint)));
}

void ChronoGramModel::saveModel(ostream & os, bool compressed) const
{
	os.write("CHGR", 4);
	writeToBinStream(os, (uint32_t)(compressed ? 4 : 3));
	writeToBinStream(os, hp);
	writeToBinStream(os, zBias);
	writeToBinStream(os, zSlope);
	auto cvocabs = vocabs;
	cvocabs.truncate(usedVocabSize());
	cvocabs.writeToFile(os);
	writeToBinStream(os, frequencies);

	subwordVocabs.writeToFile(os);
	writeToBinStream(os, subwordTable);
	writeToBinStream(os, subwordTablePtrs);

	if (compressed)
	{
		writeToBinStreamCompressed(os, in);
		writeToBinStreamCompressed(os, subwordIn);
		writeToBinStreamCompressed(os, out);
		if (hp.useSubwordOut)
		{
			writeToBinStreamCompressed(os, subwordOut);
		}
	}
	else
	{
		writeToBinStream(os, in);
		writeToBinStream(os, subwordIn);
		writeToBinStream(os, out);
		if (hp.useSubwordOut)
		{
			writeToBinStream(os, subwordOut);
		}
	}

	writeToBinStream(os, timePadding);
	writeToBinStream(os, timePrior);
}

template<class _Istream>
ChronoGramModel ChronoGramModel::loadModel(_Istream & is)
{
	auto pos = is.tellg();
	char buf[5] = { 0, };
	is.read(buf, 4);
	if (strcmp(buf, "CHGR") == 0)
	{
		size_t version = readFromBinStream<uint32_t>(is);
		if (version <= 2)
		{
			size_t d = readFromBinStream<uint32_t>(is);
			size_t r = readFromBinStream<uint32_t>(is);
			ChronoGramModel ret;
			ret.hp.dimension = d;
			ret.hp.order = r;
			ret.zBias = readFromBinStream<float>(is);
			ret.zSlope = readFromBinStream<float>(is);
			ret.vocabs.readFromFile(is);
			size_t V = ret.vocabs.size();
			ret.in.resize(d, r * V);
			ret.out.resize(d, V);

			readFromBinStream(is, ret.frequencies);
			if (version == 1)
			{
				readFromBinStream(is, ret.in);
				readFromBinStream(is, ret.out);
			}
			else if (version == 2)
			{
				readFromBinStreamCompressed(is, ret.in);
				readFromBinStreamCompressed(is, ret.out);
			}
			readFromBinStream(is, ret.hp.zeta);
			readFromBinStream(is, ret.hp.lambda);
			readFromBinStream(is, ret.timePadding);
			ret.timePrior.resize(r);
			readFromBinStream(is, ret.timePrior);
			ret.buildTable();
			ret.normalizeWordDist();
			return ret;
		}
		else if (version <= 4)
		{
			ChronoGramModel ret(readFromBinStream<HyperParameter>(is));
			ret.zBias = readFromBinStream<float>(is);
			ret.zSlope = readFromBinStream<float>(is);
			ret.vocabs.readFromFile(is);
			size_t V = ret.vocabs.size();
			readFromBinStream(is, ret.frequencies);

			ret.subwordVocabs.readFromFile(is);
			readFromBinStream(is, ret.subwordTable);
			readFromBinStream(is, ret.subwordTablePtrs);

			ret.in.resize(ret.hp.dimension, ret.hp.order * V);
			ret.out.resize(ret.hp.dimension, V);
			ret.subwordIn.resize(ret.hp.dimension, ret.hp.order * ret.subwordVocabs.size());

			if (version == 3)
			{
				readFromBinStream(is, ret.in);
				readFromBinStream(is, ret.subwordIn);
				readFromBinStream(is, ret.out);
				if (ret.hp.useSubwordOut)
				{
					ret.subwordOut.resize(ret.hp.dimension, ret.subwordVocabs.size());
					readFromBinStream(is, ret.subwordOut);
				}
			}
			else if (version == 4)
			{
				readFromBinStreamCompressed(is, ret.in);
				readFromBinStreamCompressed(is, ret.subwordIn);
				readFromBinStreamCompressed(is, ret.out);
				if (ret.hp.useSubwordOut)
				{
					ret.subwordOut.resize(ret.hp.dimension, ret.subwordVocabs.size());
					readFromBinStreamCompressed(is, ret.subwordOut);
				}
			}
			
			readFromBinStream(is, ret.timePadding);
			ret.timePrior.resize(ret.hp.order);
			readFromBinStream(is, ret.timePrior);
			ret.buildTable();
			ret.normalizeWordDist();
			return ret;
		}
		else
		{
			throw runtime_error{ "Not supported version " + to_string(version) };
		}
	}
	else
	{
		is.seekg(pos);
		size_t d = readFromBinStream<uint32_t>(is);
		size_t r = readFromBinStream<uint32_t>(is);
		ChronoGramModel ret;
		ret.hp.dimension = d;
		ret.hp.order = r;
		ret.zBias = readFromBinStream<float>(is);
		ret.zSlope = readFromBinStream<float>(is);
		ret.vocabs.readFromFile(is);
		size_t V = ret.vocabs.size();
		ret.in.resize(d, r * V);
		ret.out.resize(d, V);

		readFromBinStream(is, ret.frequencies);
		readFromBinStream(is, ret.in);
		readFromBinStream(is, ret.out);

		try
		{
			readFromBinStream(is, ret.hp.zeta);
			readFromBinStream(is, ret.hp.lambda);
			readFromBinStream(is, ret.timePadding);
			ret.timePrior.resize(r);
			readFromBinStream(is, ret.timePrior);
		}
		catch (const exception& e)
		{
			ret.timePadding = 0;
			ret.timePrior = VectorXf::Zero(r);
			ret.timePrior[0] = 1;
		}
		ret.buildTable();
		ret.normalizeWordDist();
		return ret;
	}
}

template ChronoGramModel ChronoGramModel::loadModel<istream>(istream & is);
template ChronoGramModel ChronoGramModel::loadModel<imstream>(imstream & is);

float ChronoGramModel::getWordProbByTime(const std::string & word, float timePoint) const
{
	size_t wv = vocabs.get(word);
	if (wv >= usedVocabSize()) return 0;
	return getWordProbByTime(wv, timePoint);
}

float ChronoGramModel::getTimePrior(float timePoint) const
{
	return getTimePrior(makeCoef(hp.order, normalizedTimePoint(timePoint)));
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

ChronoGramModel::LLEvaluater ChronoGramModel::evaluateSent(const vector<string>& words,
	size_t windowLen, size_t nsQ, size_t workers, const function<float(float)>& timePrior, 
	float timePriorWeight, float ctxWeight, float ugWeight,
	bool estimateSubword) const
{
	if(!workers) workers = thread::hardware_concurrency();
	if (!pool || pool->getNumWorkers() != workers)
	{
		pool = unique_ptr<ThreadPool>(new ThreadPool(workers));
	}

	LLEvaluater ret{ *this, windowLen, nsQ, timePriorWeight, ctxWeight, ugWeight };
	ret.timePrior = timePrior;

	const size_t V = usedVocabSize();
	auto& wordIds = ret.wordIds;
	auto& coefs = ret.coefs;
	auto& ugCoefs = ret.ugCoefs;
	auto& subwordTables = ret.subwordTables;
	map<vector<uint32_t>, size_t> subwordMap;

	for (auto& w : words)
	{
		uint32_t wId = vocabs.get(w);
		if (wId >= V)
		{
			if (!hp.subwordGrams || !estimateSubword) continue;
			auto sw = getSubwordIds(w);
			auto it = subwordMap.find(sw);
			if (it == subwordMap.end())
			{
				it = subwordMap.emplace(sw, subwordMap.size() + V).first;
				subwordTables.emplace_back(sw);
			}
			wordIds.emplace_back(it->second);
		}
		else
		{
			wordIds.emplace_back(wId);
		}
	}

	const auto& calcCoef = [&](size_t x, size_t y) -> VectorXf
	{
		if (!hp.subwordGrams)
		{
			return out.col(y).transpose() * in.block(0, x * hp.order, hp.dimension, hp.order);
		}

		thread_local MatrixXf s;
		if (x < V)
		{
			s = MatrixXf::Zero(hp.dimension, hp.order);
			for (size_t p = subwordTablePtrs[x]; p < subwordTablePtrs[x + 1]; ++p)
			{
				s += subwordIn.block(0, subwordTable[p] * hp.order, hp.dimension, hp.order);
			}
			s += in.block(0, x * hp.order, hp.dimension, hp.order);
		}
		else
		{
			s = MatrixXf::Zero(hp.dimension, hp.order);
			for (auto p : subwordTables[x - V])
			{
				s += subwordIn.block(0, p * hp.order, hp.dimension, hp.order);
			}
		}
		return out.col(y).transpose() * s;
	};

	size_t n = 0;
	for (size_t i = 0; i < wordIds.size(); ++i)
	{
		auto wId = wordIds[i];
		if (wId == WordDictionary<uint32_t>::npos) continue;
		if (coefs.count(wId) == 0)
		{
			coefs[wId].n = n;
			if (nsQ)
			{
				coefs[wId].dataVec.resize((V + nsQ - 1) / nsQ * hp.order);
				vector<future<void>> futures;
				const size_t stride = pool->getNumWorkers();
				for (size_t w = 0; w < stride; ++w)
				{
					futures.emplace_back(pool->enqueue([&, wId, w, n](size_t)
					{
						for (size_t v = n + w * nsQ; v < V; v += nsQ * stride)
						{
							VectorXf c = calcCoef(wId, v);
							copy(c.data(), c.data() + hp.order, &coefs[wId].dataVec[v / nsQ * hp.order]);
						}
					}));
				}
				for (auto& f : futures) f.get();
				n = (n + 1) % nsQ;
			}
		}
		size_t jBegin = 0, jEnd = wordIds.size() - 1;
		if (i > windowLen) jBegin = i - windowLen;
		if (i + windowLen < wordIds.size()) jEnd = i + windowLen;
		for (size_t j = jBegin; j <= jEnd; ++j)
		{
			if (i == j) continue;
			auto v = wordIds[j];
			if (v >= V) continue;
			if (coefs[wId].dataMap.count(v)) continue;

			VectorXf c = calcCoef(wId, v);
			coefs[wId].dataMap[v] = { c.data(), c.data() + hp.order };
		}

		if (hp.ugWeight && !ugCoefs.count(wId))
		{
			MatrixXf s;
			if (wId < V)
			{
				s = MatrixXf::Zero(hp.ugPartDimension, hp.order);
				if (hp.subwordGrams)
				{
					for (size_t p = subwordTablePtrs[wId]; p < subwordTablePtrs[wId + 1]; ++p)
					{
						s += subwordIn.block(0, subwordTable[p] * hp.order, hp.ugPartDimension, hp.order);
					}
				}
				s += in.block(0, wId * hp.order, hp.ugPartDimension, hp.order);
			}
			else
			{
				s = MatrixXf::Zero(hp.ugPartDimension, hp.order);
				for (auto p : subwordTables[wId - V])
				{
					s += subwordIn.block(0, p * hp.order, hp.ugPartDimension, hp.order);
				}
			}
			ugCoefs[wId] = s.transpose() * s;
		}
	}

	return ret;
}


pair<float, float> ChronoGramModel::predictSentTime(const vector<string>& words,
	size_t windowLen, size_t nsQ, const function<float(float)>& timePrior, float timePriorWeight,
	size_t initStep, float threshold, std::vector<float>* llOutput) const
{
	auto evaluator = evaluateSent(words, windowLen, nsQ, 0, timePrior, timePriorWeight);
	constexpr uint32_t uScale = 0x80000000;
	map<uint32_t, pair<float, float>> lls;
	float maxLL = -INFINITY;
	uint32_t maxP = 0;
	if (llOutput) llOutput->clear();
	for (size_t i = 0; i <= initStep; ++i)
	{
		auto t = evaluator.fg(i / (float)initStep * (1 - timePadding * 2) + timePadding);
		auto m = (uint32_t)(uScale * (float)i / initStep);
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
		if (it->first - prevIt->first < (uint32_t)(uScale * threshold)) continue;
		if (prevIt->second.second < 0) continue;
		if (it->second.second > 0) continue;
		auto m = (prevIt->first + it->first) / 2;
		auto t = evaluator.fg(m / (float)uScale * (1 - timePadding * 2) + timePadding);
		lls.emplace(m, make_pair(get<0>(t), get<1>(t)));
		it = prevIt;
		if (get<0>(t) > maxLL)
		{
			maxP = m;
			maxLL = get<0>(t);
		}
	}

	return make_pair(unnormalizedTimePoint(maxP / (float)uScale * (1 - timePadding * 2) + timePadding), maxLL);
}

vector<ChronoGramModel::EvalResult> ChronoGramModel::evaluate(const function<ReadResult()>& reader,
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
		auto r = reader();
		if (r.stop) break;
		unique_lock<mutex> l{ readMtx };
		readCnd.wait(l, [&]() { return workers.getNumEnqued() < workers.getNumWorkers() * 4; });
		consume();
		workers.enqueue([&, id](size_t tid, float time, vector<string> words)
		{
			vector<float> lls;
			auto p = predictSentTime(words, windowLen, nsQ, timePrior, timePriorWeight, initStep, threshold, &lls);
			lock_guard<mutex> l{ writeMtx };
			res[id] = EvalResult{ time, p.first, p.second, p.second / 2 / windowLen / words.size(),
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

float ChronoGramModel::LLEvaluater::operator()(float normalizedTimePoint) const
{
	const size_t N = wordIds.size(), V = tgm.usedVocabSize();
	auto tCoef = makeCoef(tgm.hp.order, normalizedTimePoint);
	auto defaultPrior = [&](float)->float
	{
		return log(TempAct{}(pow(tgm.timePrior.dot(tCoef), 2) / 2).first + 1e-5f);
	};
	float pa = max(tgm.getTimePrior(tCoef), 1e-5f);
	float ll = (timePrior && timePriorWeight ? timePrior : defaultPrior)(tgm.unnormalizedTimePoint(normalizedTimePoint)) * timePriorWeight;
	unordered_map<uint32_t, uint32_t> count;

	MatrixXf ttCoef = tCoef * tCoef.transpose();

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
			if (y >= V) continue;
			float d = (tCoef.array() * Map<const ArrayXf>{ cx.get(y, nsQ, tgm.hp.order), tgm.hp.order }).sum();
			ll += logsigmoid(d) * ctxWeight;
			count[x]++;
		}

		if (tgm.hp.ugWeight)
		{
			ll += log(TempAct{}((ugCoefs.find(x)->second.array() * ttCoef.array()).sum()).first + 1e-12f) * tgm.hp.ugWeight * ugWeight;
		}
	}

	if (nsQ)
	{
		for (auto& p : count)
		{
			thread_local VectorXf tVec;
			auto& cx = coefs.find(p.first)->second;
			float nll = 0;
			float denom = 0;
			tVec = Map<const MatrixXf>(cx.dataVec.data(), tgm.hp.order, (V - cx.n + nsQ - 1) / nsQ).transpose() * tCoef;
			//tVec = ((-tVec.array().tanh() + 1) / 2).log();
			for (size_t j = cx.n, idx = 0; j < V; j += nsQ, ++idx)
			{
				//float d = (tCoef.array() * Map<const ArrayXf>{ cx.get(j, nsQ, tgm.hp.order), tgm.hp.order }).sum();
				float d = tVec[idx];
				nll += tgm.unigramDist[j] * logsigmoid(-d);
				//nll += tgm.unigramDist[j] * tVec[idx];
				denom += tgm.unigramDist[j];
			}
			ll += nll / denom * tgm.hp.negativeSamples * p.second;
		}
	}
	return ll;
}

tuple<float, float> ChronoGramModel::LLEvaluater::fg(float normalizedTimePoint) const
{
	const size_t N = wordIds.size(), V = tgm.unigramDist.size();
	auto tCoef = makeCoef(tgm.hp.order, normalizedTimePoint), tDCoef = makeDCoef(tgm.hp.order, normalizedTimePoint);
	auto defaultPrior = [&](float)->float
	{
		return log(TempAct{}(pow(tgm.timePrior.dot(tCoef), 2) / 2).first + 1e-5f);
	};
	float pa = max(tgm.getTimePrior(tCoef), 1e-5f);
	float dot = tgm.timePrior.dot(tCoef);
	float ddot = tgm.timePrior.middleRows(1, tgm.hp.order - 1).dot(tDCoef);
	float ll = (timePrior ? timePrior : defaultPrior)(tgm.unnormalizedTimePoint(normalizedTimePoint)) * timePriorWeight,
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
			float d = (tCoef.array() * Map<const ArrayXf>{ cx.get(y, nsQ, tgm.hp.order), (Eigen::Index)tgm.hp.order }).sum();
			ll += logsigmoid(d) * (1 - tgm.hp.zeta);
			float dd = (tDCoef.array() * Map<const ArrayXf>{ cx.get(y, nsQ, tgm.hp.order) + 1, (Eigen::Index)tgm.hp.order - 1 }).sum(), sd;
			dll += dd * (sd = sigmoid(-d)) * (1 - tgm.hp.zeta);
			count[x]++;
		}

		auto v = tgm.makeTimedVector(x, tCoef);
		float sqn = v.squaredNorm() / 2 * tgm.hp.lambda * pa, sd;
		ll += log(1 - exp(-sqn) + 1e-5f) * tgm.hp.zeta;
		dll += v.dot(tgm.in.block(0, x * tgm.hp.order + 1, tgm.hp.dimension, tgm.hp.order - 1) * tDCoef) * tgm.hp.lambda * pa / (sd = exp(sqn) - 1 + 1e-5f) * tgm.hp.zeta;
	}

	if (nsQ)
	{
		for (auto& p : count)
		{
			auto& cx = coefs.find(p.first)->second;
			float nll = 0, dnll = 0;
			float denom = 0;
			for (size_t j = cx.n; j < V; j += nsQ)
			{
				float d = (tCoef.array() * Map<const ArrayXf>{ cx.get(j, nsQ, tgm.hp.order), tgm.hp.order }).sum();
				nll += tgm.unigramDist[j] * logsigmoid(-d);
				float dd = (tDCoef.array() * Map<const ArrayXf>{ cx.get(j, nsQ, tgm.hp.order) + 1, tgm.hp.order - 1 }).sum(), sd;
				dnll += -tgm.unigramDist[j] * dd * (sd = sigmoid(d));
				denom += tgm.unigramDist[j];
			}
			ll += nll / denom * tgm.hp.negativeSamples * p.second * (1 - tgm.hp.zeta);
			dll += dnll / denom * tgm.hp.negativeSamples * p.second * (1 - tgm.hp.zeta);
		}
	}
	return make_tuple(ll, dll);
}

template void ChronoGramModel::train<false>(const std::function<ResultReader()>&, size_t, size_t, float, float, size_t, float, size_t, const ReportCallback& reportCallback);
template void ChronoGramModel::train<true>(const std::function<ResultReader()>&, size_t, size_t, float, float, size_t, float, size_t, const ReportCallback& reportCallback);
template void ChronoGramModel::trainFromGNgram<false>(const std::function<GNgramResultReader()>&, uint64_t, size_t, float, float, size_t, float, size_t, const ReportCallback& reportCallback);
template void ChronoGramModel::trainFromGNgram<true>(const std::function<GNgramResultReader()>&, uint64_t, size_t, float, float, size_t, float, size_t, const ReportCallback& reportCallback);
