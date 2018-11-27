#pragma once

#include "gamma.h"

inline float sigmoid(float x)
{
	return 1 / (1 + exp(-x));
}

inline float logsigmoid(float x)
{
	return -log(1 + exp(-x));
}

inline float mean_beta(float a, float b)
{
	return a / (a + b);
}

inline float meanlog_beta(float a, float b)
{
	return DIGAMMA(a) - DIGAMMA(a + b);
}

inline float mean_mirror(float a, float b)
{
	return mean_beta(b, a);
}

inline float meanlog_mirror(float a, float b)
{
	return meanlog_beta(b, a);
}
