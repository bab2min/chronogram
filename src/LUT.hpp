#pragma once

template<class MathFunction, class Precision, size_t N, size_t S, size_t M, size_t T, size_t L, size_t U>
class LUT3
{
protected:
	Precision* points;
	const Precision P = 1.0 / S;
	const Precision Q = 1.0 / T;
	const Precision R = 1.0 / U;
	LUT3()
	{
		MathFunction fun;
		points = new Precision[N + M + L];
		for (size_t i = 0; i < N; i++)
		{
			points[i] = fun(i ? i*P : 0.0001);
		}
		for (size_t i = 0; i < M; i++)
		{
			points[i + N] = fun(i*Q + N*P);
		}
		for (size_t i = 0; i < L; i++)
		{
			points[i + N + M] = fun(i*R + N*P + M*Q);
		}
	}
	~LUT3()
	{
		delete[] points;
	}

	Precision _get(Precision x) const
	{
		if (x < 0) return NAN;
		if (x < MathFunction::smallThreshold) return MathFunction{}.forSmall(x);
		if (x >= N / S + M / T + L / U) return MathFunction{}.forLarge(x);
		size_t idx;
		Precision a;
		Precision nx = x;
		if (x < N*P)
		{
			idx = (size_t)(nx / P);
			a = (nx - idx*P) / P;
		}
		else
		{
			nx -= N*P;
			if (nx < M*Q)
			{
				idx = (size_t)(nx / Q);
				a = (nx - idx*Q) / Q;
				idx += N;
			}
			else
			{
				nx -= M*Q;
				idx = (size_t)(nx / R);
				a = (nx - idx*R) / R;
				idx += N + M;
			}
		}
		return points[idx] + a * (points[idx + 1] - points[idx]);
	}
public:
	static const LUT3& getInst()
	{
		static LUT3 lg;
		return lg;
	}

	static Precision get(Precision x)
	{
		return getInst()._get(x);
	}
};

