#pragma once

#include <chrono>

class Timer
{
public:
	std::chrono::high_resolution_clock::time_point point;
	Timer()
	{
		reset();
	}
	void reset()
	{
		point = std::chrono::high_resolution_clock::now();
	}

	double getElapsed() const
	{
		return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - point).count();
	}
};
