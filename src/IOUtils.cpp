#include <Eigen/Dense>
#include "IOUtils.h"

using namespace std;

constexpr int VL_PRECISION = 1 << 12;
constexpr int VL_1B = 1 << 6;
constexpr int VL_2B = 1 << 14;

void writeFloatVL(ostream& os, float f)
{
	int n = f * VL_PRECISION;
	if (-VL_1B <= n && n < VL_1B)
	{
		uint8_t w = (uint8_t)n & 0x7F;
		os.write((const char*)&w, 1);
	}
	else if (-VL_2B <= n && n < VL_2B)
	{
		uint16_t un = (uint16_t)n & 0x7FFF;
		uint8_t w[2] = { (uint8_t)(0x80 | (un >> 8)), (uint8_t)(un & 0xFF)};
		os.write((const char*)w, 2);
	}
	else if(n >= VL_2B)
	{
		uint8_t w[2] = { (uint8_t)(0xFF ^ 0x40), 0xFF };
		os.write((const char*)w, 2);
	}
	else
	{
		uint8_t w[2] = { (uint8_t)(0x80 | 0x40), 0x00 };
		os.write((const char*)w, 2);
	}
}

float readFloatVL(istream& is)
{
	uint8_t w[2] = { 0, };
	is.read((char*)w, 1);
	if (w[0] & 0x80)
	{
		is.read((char*)w + 1, 1);
		w[0] ^= 0x80;
		w[0] |= (w[0] & 0x40) << 1;
		uint16_t un = (w[0] << 8) | w[1];
		return (int16_t)un / (float)VL_PRECISION;
	}
	else
	{
		w[0] |= (w[0] & 0x40) << 1;
		return (int8_t)w[0] / (float)VL_PRECISION;
	}
}
