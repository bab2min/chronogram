#pragma once

inline float sigmoid(float x)
{
	return 1 / (1 + exp(-x));
}

inline float logsigmoid(float x)
{
	return -log(1 + exp(-x));
}
