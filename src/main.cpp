// AdaGram-Cpp.cpp : Defines the entry point for the console application.
//

#include <vector>
#include <array>
#include <map>
#include <algorithm>
#include <unordered_map>
#include <numeric>
#include <string>
#include <iostream>
#include <sstream>
#include <iterator>
#include <fstream>
#include <random>

#include "TimeGramModel.h"

using namespace std;

bool posSelect(const string& o)
{
	if (o.find("/NN") != string::npos) return true;
	if (o.find("/VA") != string::npos) return true;
	if (o.find("/VV") != string::npos) return true;
	if (o.find("/MM") != string::npos) return true;
	if (o.find("/MAG") != string::npos) return true;
	if (o.find("/XR") != string::npos) return true;
	if (o.find("/SL") != string::npos) return true;
	return false;
}

string unifyNoun(const string& o)
{
	if (o.find("/NN") != string::npos) return o.substr(0, o.size() - 1);
	return o;
}


int main(int argc, const char** argv)
{
	TimeGramModel tgm{ 100, 5, 1e-4, 5 };
	if (1)
	{
		Timer timer;
		ifstream ifs{ argv[1] };
		
		tgm.buildVocab([&ifs](size_t id)
		{
			TimeGramModel::ReadResult rr;
			string line;
			if (id == 0)
			{
				ifs.clear();
				ifs.seekg(0);
			}

			while (getline(ifs, line))
			{
				istringstream iss{ line };
				istream_iterator<string> iBegin{ iss }, iEnd{};
				copy(iBegin, iEnd, back_inserter(rr.words));
				if (rr.words.empty()) continue;
				return rr;
			}
			rr.stop = true;
			return rr;
		});

		mt19937_64 gen{ 1 };

		tgm.train([&ifs, &gen](size_t id)
		{
			TimeGramModel::ReadResult rr;
			rr.timePoint = generate_canonical<float, 24>(gen);
			string line;
			if (id == 0)
			{
				ifs.clear();
				ifs.seekg(0);
			}

			while (getline(ifs, line))
			{
				istringstream iss{ line };
				istream_iterator<string> iBegin{ iss }, iEnd{};
				copy(iBegin, iEnd, back_inserter(rr.words));
				if (rr.words.empty()) continue;
				return rr;
			}
			rr.stop = true;
			return rr;
		}, 0, 4, .025f, 1000, 5);

		cout << "Finished in " << timer.getElapsed() << " sec" << endl;
		ofstream ofs{ argv[2], ios_base::binary };
		tgm.saveModel(ofs);
	}
	else
	{
		ifstream ifs{ argv[2], ios_base::binary };
		tgm = TimeGramModel::loadModel(ifs);

	}

	string line;
	while (cout << ">> ", getline(cin, line))
	{
		vector<string> positives, negatives;
		istringstream iss{ line };
		istream_iterator<string> wBegin{ iss }, wEnd{};
		float timePoint = 0;
		bool sign = false;
		for (; wBegin != wEnd; ++wBegin)
		{
			auto word = *wBegin;
			if (word == "-")
			{
				sign = true;
			}
			else if (word == "+")
			{
				sign = false;
			}
			else if (word[0] == '@')
			{
				timePoint = stof(word.substr(1));
			}
			else
			{
				(sign ? negatives : positives).emplace_back(word);
				sign = false;
			}
		}
		
		cout << "================" << endl;
		for (auto& p : tgm.mostSimilar(positives, negatives, timePoint, 20))
		{
			cout << get<0>(p) << '\t' << get<1>(p) << endl;
		}
		cout << endl;
	}
    return 0;
}

