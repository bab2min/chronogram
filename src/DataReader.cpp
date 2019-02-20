#include <iterator>
#include "DataReader.h"
#include "IOUtils.h"

using namespace std;

ChronoGramModel::ReadResult MultipleReader::operator()(size_t id)
{
	ChronoGramModel::ReadResult rr;
	string line;
	if (id == 0)
	{
		currentId = 0;
		ifs = ifstream{ files[currentId] };
	}

	while (currentId < files.size())
	{
		while (getline(ifs, line))
		{
			istringstream iss{ line };
			istream_iterator<string> iBegin{ iss }, iEnd{};
			string time = *iBegin++;
			rr.timePoint = stof(time);
			copy(iBegin, iEnd, back_inserter(rr.words));
			if (rr.words.empty()) continue;
			return rr;
		}
		if (++currentId >= files.size()) break;
		ifs = ifstream{ files[currentId] };
	}
	rr.stop = true;
	return rr;
}

size_t decodeVLE(const uint8_t* &p)
{
	size_t ret = 0;
	for (size_t n = 0; n < 5; ++n)
	{
		ret <<= 7;
		ret |= *p & 0x7F;
		if ((*p++ & 0x80) == 0) break;
	}
	return ret;
}

uint16_t decode16(const uint8_t* &p)
{
	uint16_t ret = p[0] | (p[1] << 8);
	p += 2;
	return ret;
}

ChronoGramModel::GNgramReadResult GNgramBinaryReader::operator()(size_t id)
{
	if (id == 0)
	{
		ifs.clear();
		ifs.seekg(0);
	}

	ChronoGramModel::GNgramReadResult ret;
	uint8_t buf[16384];
	try
	{
		size_t len = readFromBinStream<uint16_t>(ifs);
		if (!ifs.read((char*)buf, len)) return ret;
		const uint8_t* p = buf;
		for (size_t i = 0; i < 5; ++i)
		{
			ret.ngram[i] = decodeVLE(p) - 1;
		}
		len = decode16(p);
		for (size_t i = 0; i < len; ++i)
		{
			float year = decode16(p);
			uint32_t cnt = decode16(p);
			ret.yearCnt.emplace_back(year, cnt);
		}
		return ret;
	}
	catch (ios_base::failure)
	{
		return ret;
	}
}
