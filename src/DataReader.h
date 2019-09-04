#pragma once
#include <fstream>
#include <memory>
#include "ChronoGramModel.h"

class MultipleReader
{
	std::vector<std::string> files;
	size_t currentId = 0;
	std::unique_ptr<std::ifstream> ifs;
public:
	MultipleReader(const std::vector<std::string>& _files) : files(_files), ifs(new std::ifstream{_files[0]})
	{
	}

	ChronoGramModel::ReadResult operator()();
	static std::function<ChronoGramModel::ResultReader()> factory(const std::vector<std::string>& _files);
};

class GNgramBinaryReader
{
	std::ifstream ifs;
public:
	GNgramBinaryReader(const std::string& filename) : ifs(filename, std::ios_base::binary)
	{
	}

	ChronoGramModel::GNgramReadResult operator()();
	static std::function<ChronoGramModel::GNgramResultReader()> factory(const std::string& _file);
};
