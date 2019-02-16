#pragma once
#include <fstream>
#include "ChronoGramModel.h"

struct MultipleReader
{
	std::vector<std::string> files;
	size_t currentId = 0;
	std::ifstream ifs;

	MultipleReader(const std::vector<std::string>& _files) : files(_files)
	{
		ifs = std::ifstream{ files[currentId] };
	}

	ChronoGramModel::ReadResult operator()(size_t id);
};

struct GNgramBinaryReader
{
	std::ifstream ifs;

	GNgramBinaryReader(std::string filename) : ifs(filename, std::ios_base::binary)
	{
	}

	ChronoGramModel::GNgramReadResult operator()(size_t id);
};