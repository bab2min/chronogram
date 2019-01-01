#pragma once
#include <array>

inline float sigmoid(float x)
{
	return 1 / (1 + exp(-x));
}

struct F_logsigmoid
{
	float operator()(float x) { return -log(1 + exp(-x)); }
	float forLarge(float x) { return -0.f; }
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
	if (x >= 0) return SimpleLUT<F_logsigmoid, 32 * 128, 128>::get(x);
	return SimpleLUT<F_logsigmoid, 32 * 128, 128>::get(-x) + x;
}
