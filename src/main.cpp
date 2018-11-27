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


int main()
{
	TimeGramModel agm{ 300, 3, 1e-4, 5 };
	if (0)
	{
		Timer timer;
		ifstream ifs{ "D:/namu_tagged.txt" };
		
		cout << "Finished in " << timer.getElapsed() << " sec" << endl;
		ofstream ofs{ "namu_subsampling.mdl", ios_base::binary };
		agm.saveModel(ofs);
	}
	else
	{
		ifstream ifs{ "namu_subsampling.mdl", ios_base::binary };
		agm = TimeGramModel::loadModel(ifs);

	}

	string line;
	while (cout << ">> ", getline(cin, line))
	{
		vector<string> positives, negatives;
		istringstream iss{ line };
		istream_iterator<string> wBegin{ iss }, wEnd{};
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
			else
			{
				(sign ? negatives : positives).emplace_back(word);
				sign = false;
			}
		}
		
		cout << "================" << endl;
		for (auto& p : agm.mostSimilar(positives, negatives, 20))
		{
			cout << get<0>(p) << '\t' << get<1>(p) << endl;
		}
		cout << endl;
	}
    return 0;
}

