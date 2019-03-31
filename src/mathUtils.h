#pragma once
#include <array>

inline float sigmoid(float x)
{
	return 1 / (1 + exp(-x));
}

struct F_logsigmoid
{
	double operator()(double x) { return -log(1 + exp(-x)); }
	double forLarge(double x) { return -0.f; }
};

template<class _Func, size_t N, size_t S>
class SimpleLUT
{
protected:
	std::array<float, N> points;
	static constexpr float P = 1.f / S;
	SimpleLUT()
	{
		_Func fun;
		for (size_t i = 0; i < N; i++)
		{
			points[i] = fun(i * P);
		}
	}

	float _get(float x) const
	{
		size_t idx = (size_t)(x * S);
		if (idx >= N) return _Func{}.forLarge(x);
		return points[idx];
	}
public:
	static const SimpleLUT& getInst()
	{
		static SimpleLUT lg;
		return lg;
	}

	static float get(float x)
	{
		return getInst()._get(x);
	}
};

inline float logsigmoid(float x)
{
	if (x >= 0) return SimpleLUT<F_logsigmoid, 32 * 1024, 1024>::get(x);
	return SimpleLUT<F_logsigmoid, 32 * 1024, 1024>::get(-x) + x;
}

inline float integratedChebyshevT(size_t n)
{
	if (n % 2) return 0;
	return 2.f / (1.f - n * n);
}

inline float integratedSqChebyshevT(size_t n)
{
	return 1 - 1.f / (4 * n * n - 1.f);
}

inline float integratedChebyshevTT(size_t n, size_t m)
{
	return (integratedChebyshevT(n + m) + integratedChebyshevT(std::max(n, m) - std::min(n, m))) / 2;
}

template<typename _OutTy, typename _RandIter>
_OutTy correlPearson(_RandIter xFirst, _RandIter xLast, _RandIter yFirst)
{
	_OutTy xMean = 0, yMean = 0, xSqMean = 0, ySqMean = 0, xyMean = 0;
	size_t cnt = 0;
	for (auto x = xFirst, y = yFirst; x != xLast; ++x, ++y, ++cnt)
	{
		xMean += *x;
		yMean += *y;
		xSqMean += *x * *x;
		ySqMean += *y * *y;
		xyMean += *x * *y;
	}
	xMean /= cnt;
	yMean /= cnt;
	xSqMean /= cnt;
	ySqMean /= cnt;
	xyMean /= cnt;
	return (xyMean - xMean * yMean) / sqrt((xSqMean - xMean * xMean) * (ySqMean - yMean * yMean));
}

template<typename _InputIter, typename _OutputIter>
void rankAvg(_InputIter srcFirst, _InputIter srcLast, _OutputIter destFirst)
{
	size_t len = std::distance(srcFirst, srcLast);
	std::vector<size_t> idx(len);
	std::iota(idx.begin(), idx.end(), 0);
	std::sort(idx.begin(), idx.end(), [&](auto l, auto r)
	{
		return srcFirst[l] < srcFirst[r];
	});
	size_t rank = 1;
	for (size_t i = 0; i < len; )
	{
		size_t n = 1;
		while (i + n < len && srcFirst[idx[i + n - 1]] == srcFirst[idx[i + n]]) ++n;

		for (size_t j = 0; j < n; ++j)
		{
			destFirst[idx[i + j]] = rank + (n - 1) * .5;
		}
		rank += n;
		i += n;
	}
}

template<typename _OutTy, typename _RandIter>
_OutTy correlSpearman(_RandIter xFirst, _RandIter xLast, _RandIter yFirst)
{
	size_t len = std::distance(xFirst, xLast);
	std::vector<_OutTy> xRank(len), yRank(len);
	rankAvg(xFirst, xLast, xRank.begin());
	rankAvg(yFirst, std::next(yFirst, len), yRank.begin());
	return correlPearson<_OutTy>(xRank.begin(), xRank.end(), yRank.begin());
}
