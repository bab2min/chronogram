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

#include "cxxopts.hpp"
#include "TimeGramModel.h"

using namespace std;

struct Args
{
	string input, load, save;
	int worker = 0, window = 4, dimension = 100;
	int order = 5, epoch = 1, negative = 5;
	int batch = 10000;
};

int main(int argc, char* argv[])
{
	Args args;
	try
	{
		cxxopts::Options options("chronogram", "Diachronic Word Embeddings");
		/*options
			.positional_help("[mode]")
			.show_positional_help();
		*/
		options.add_options()
			/*("mode", "", cxxopts::value<string>(), "train, load")*/
			("i,input", "Input File", cxxopts::value<string>(), "Input file path that contains documents per line")
			("l,load", "Load Model File", cxxopts::value<string>(), "Model file path to be loaded")
			("v,save", "Save Model File", cxxopts::value<string>(), "Model file path to be saved")
			("h,help", "Help")
			("version", "Version")

			("w,worker", "Number of Workes", cxxopts::value<int>(), "The number of workers(thread) for inferencing model, default value is 0 which means the number of cores in system")
			("W,window", "Size of Window", cxxopts::value<int>())
			("d,dimension", "Embedding Dimension", cxxopts::value<int>())
			("r,order", "Order of Chebyshev Polynomial", cxxopts::value<int>())
			("e,epoch", "Number of Epoch", cxxopts::value<int>())
			("n,negative", "Negative Sampling Size", cxxopts::value<int>())
			("b,batch", "Batch Docs Size", cxxopts::value<int>())
			;

		//options.parse_positional({ "model", "input", "topic" });

		try
		{
			auto result = options.parse(argc, argv);

			if (result.count("version"))
			{
				cout << "v0.1" << endl;
				return 0;
			}
			if (result.count("help"))
			{
				cout << options.help({ "" }) << endl;
				return 0;
			}


#define READ_OPT(P, TYPE) if (result.count(#P)) args.P = result[#P].as<TYPE>()
#define READ_OPT2(P, Q, TYPE) if (result.count(#P)) args.Q = result[#P].as<TYPE>()

			READ_OPT(load, string);
			READ_OPT(save, string);
			READ_OPT(input, string);

			READ_OPT(worker, int);
			READ_OPT(window, int);
			READ_OPT(dimension, int);
			READ_OPT(order, int);
			READ_OPT(epoch, int);
			READ_OPT(negative, int);
			READ_OPT(batch, int);
			
			if (args.load.empty() && args.input.empty())
			{
				throw cxxopts::OptionException("'input' or 'load' should be specified.");
			}
		}
		catch (const cxxopts::OptionException& e)
		{
			cout << "error parsing options: " << e.what() << endl;
			cout << options.help({ "" }) << endl;
			return -1;
		}

	}
	catch (const cxxopts::OptionException& e)
	{
		cout << "error parsing options: " << e.what() << endl;
		return -1;
	}

	cout << "Dimension: " << args.dimension << "\tOrder: " << args.order << "\tNegative Sampling: " << args.negative << endl;
	cout << "Workers: " << args.worker << "\tBatch: " << args.batch << "\tEpochs: " << args.epoch << endl;
	TimeGramModel tgm{ (size_t)args.dimension, (size_t)args.order, 1e-4, (size_t)args.negative };
	if (!args.input.empty())
	{
		cout << "Training Input: " << args.input << endl;
		Timer timer;
		ifstream ifs{ args.input };
		const auto reader = [&ifs](size_t id)
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
				string time = *iBegin++;
				rr.timePoint = stof(time);
				copy(iBegin, iEnd, back_inserter(rr.words));
				if (rr.words.empty()) continue;
				return rr;
			}
			rr.stop = true;
			return rr;
		};
		tgm.buildVocab(reader);
		cout << "Vocab Size: " << tgm.getVocabs().size() << endl;
		tgm.train(reader, args.worker, args.window, .025f, args.batch, args.epoch);

		cout << "Finished in " << timer.getElapsed() << " sec" << endl;
		if (!args.save.empty())
		{
			cout << "Saving Model: " << args.save << endl;
			ofstream ofs{ args.save, ios_base::binary };
			tgm.saveModel(ofs);
		}
	}
	else if(!args.load.empty())
	{
		cout << "Loading Model: " << args.load << endl;
		ifstream ifs{ args.load, ios_base::binary };
		tgm = TimeGramModel::loadModel(ifs);

	}

	string line;
	while (cout << ">> ", getline(cin, line))
	{
		vector<string> positives, negatives;
		istringstream iss{ line };
		istream_iterator<string> wBegin{ iss }, wEnd{};
		float timePoint = tgm.getMinPoint();
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

