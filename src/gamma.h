#pragma once

#include <cmath>
#include "LUT.hpp"

#define M_PIf 3.14159265f
#define M_GAMMAf 0.577215664f
#define M_LN2f 0.693147180f

inline float digammaf(float x)
{
	if (x < 0.0f)
		return digammaf(1.0f - x) + M_PIf / tanl(M_PIf*(1.0f - x));
	else if (x < 1.0f)
		return digammaf(1.0f + x) - 1.0f / x;
	else if (x == 1.0f)
		return -M_GAMMAf;
	else if (x == 2.0f)
		return 1.0L - M_GAMMAf;
	else if (x == 3.0f)
		return 1.5L - M_GAMMAf;
	else if (x > 3.0f)
		return 0.5f*(digammaf(x / 2.0f) + digammaf((x + 1.0f) / 2.0f)) + M_LN2f;
	else
	{
		static float Kncoe[] = { .304591985f,
			.720379774f, -.124549592f,
			.277694573e-1f, -.677623714e-2f,
			.172387551e-2f, -.448176990e-3f,
			.117936600e-3f, -.312538942e-4f,
			.831739970e-5f, -.221914276e-5f,
			.593022667e-6f, -.158630511e-6f,
			.424592039e-7f, -.113691296e-7f,
			.304502217e-8f, -.815684550e-9f };

		float Tn_1 = 1.0L;
		float Tn = x - 2.0L;
		float resul = Kncoe[0] + Kncoe[1] * Tn;

		x -= 2.0L;

		for (int n = 2; n < sizeof(Kncoe) / sizeof(float); n++)
		{
			const float Tn1 = 2.0L * x * Tn - Tn_1;
			resul += Kncoe[n] * Tn1;
			Tn_1 = Tn;
			Tn = Tn1;
		}
		return resul;
	}
}


struct F_digamma
{
	float operator()(float x)
	{
		return digammaf(x);
	}
	static constexpr float smallThreshold = 0.1;
	float forSmall(float x) { return log(x + 2) - 0.5 / (x + 2) - 1 / 12.0 / pow(x + 2, 2) - 1 / (x + 1) - 1 / x; }
	float forLarge(float x) { return log(x) - 0.5/x - 1 / 12.0 / pow(x, 2); }
};

typedef LUT3<F_digamma, float, 1 * 1024, 1024, 100 * 64, 64, 1000 * 4, 4> LUT_digamma;

#define DIGAMMA(x) LUT_digamma::get(x)
