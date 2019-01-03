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
	string input, load, save, eval, result;
	int worker = 0, window = 4, dimension = 100;
	int order = 5, epoch = 1, negative = 5;
	int batch = 10000, minCnt = 10;
	int report = 100000;
	int nsQ = 8, initStep = 10;
	float eta = 1.f;
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
			("eval", "Evaluation set File", cxxopts::value<string>(), "Evaluation set file path")
			("result", "Evaluation Result File", cxxopts::value<string>(), "Evaluation result file path")
			("h,help", "Help")
			("version", "Version")

			("w,worker", "Number of Workes", cxxopts::value<int>(), "The number of workers(thread) for inferencing model, default value is 0 which means the number of cores in system")
			("W,window", "Size of Window", cxxopts::value<int>())
			("d,dimension", "Embedding Dimension", cxxopts::value<int>())
			("r,order", "Order of Chebyshev Polynomial", cxxopts::value<int>())
			("e,epoch", "Number of Epoch", cxxopts::value<int>())
			("n,negative", "Negative Sampling Size", cxxopts::value<int>())
			("b,batch", "Batch Docs Size", cxxopts::value<int>())
			("t,minCnt", "Min Count Threshold of Word", cxxopts::value<int>())
			("report", "", cxxopts::value<int>())
			("nsQ", "", cxxopts::value<int>())
			("initStep", "", cxxopts::value<int>())
			("eta", "", cxxopts::value<float>())
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
			READ_OPT(eval, string);
			READ_OPT(result, string);

			READ_OPT(worker, int);
			READ_OPT(window, int);
			READ_OPT(dimension, int);
			READ_OPT(order, int);
			READ_OPT(epoch, int);
			READ_OPT(negative, int);
			READ_OPT(batch, int);
			READ_OPT(minCnt, int);
			READ_OPT(nsQ, int);
			READ_OPT(initStep, int);

			READ_OPT(eta, float);
			
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
	cout << "Eta: " << args.eta << endl;
	TimeGramModel tgm{ (size_t)args.dimension, (size_t)args.order, 1e-4, (size_t)args.negative, args.eta };
	if (!args.load.empty())
	{
		cout << "Loading Model: " << args.load << endl;
		ifstream ifs{ args.load, ios_base::binary };
		tgm = TimeGramModel::loadModel(ifs);
	}
	else if (!args.input.empty())
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
		tgm.buildVocab(reader, args.minCnt);
		cout << "MinCnt: " << args.minCnt << "\tVocab Size: " << tgm.getVocabs().size() << endl;
		tgm.train(reader, args.worker, args.window, .025f, args.batch, args.epoch);

		cout << "Finished in " << timer.getElapsed() << " sec" << endl;
		if (!args.save.empty())
		{
			cout << "Saving Model: " << args.save << endl;
			ofstream ofs{ args.save, ios_base::binary };
			tgm.saveModel(ofs);
		}
	}

	if (!args.eval.empty())
	{
		cout << "Evaluating Time Prediction: " << args.eval << endl;
		Timer timer;
		ifstream ifs{ args.eval };
		if (args.result.empty()) args.result = args.eval + ".result";
		ofstream ofs{ args.result };
		string line;
		float avgErr = 0;
		size_t n = 0;
		while (getline(ifs, line))
		{
			istringstream iss{ line };
			istream_iterator<string> iBegin{ iss }, iEnd{};
			string time = *iBegin++;
			float timePoint = stof(time);
			vector<string> words{ iBegin, iEnd };
			auto llp = tgm.predictSentTime(words, args.window, args.nsQ, args.initStep);
			float normalizedErr = tgm.normalizedTimePoint(timePoint) - tgm.normalizedTimePoint(llp.first);
			ofs << timePoint << "\t" << llp.first << "\t" << llp.second
				<< "\t" << llp.second / (args.window * 2 * words.size())
				<< "\t" << timePoint - llp.first << "\t" << normalizedErr << endl;
			avgErr += pow(normalizedErr, 2);
			n++;
		}
		avgErr /= n;
		cout << "== Evaluating Result ==" << endl;
		cout << "Running Time: " << timer.getElapsed() << "s" << endl;
		cout << "Total Docs: " << n << endl;
		cout << "Avg Squared Error: " << avgErr << endl;
		cout << "Avg Error: " << sqrt(avgErr) << endl;

		ofs << "Running Time: " << timer.getElapsed() << "s" << endl;
		ofs << "Total Docs: " << n << endl;
		ofs << "Avg Squared Error: " << avgErr << endl;
		ofs << "Avg Error: " << sqrt(avgErr) << endl;
	}

	string line;
	while (cout << ">> ", getline(cin, line))
	{
		if (line[0] == '~') // calculate avg similarity per period
		{
			if (line[1] == '~') // find shortest arc length
			{
				cout << "==== Shortest Arc Length ====" << endl;
				vector<pair<string, float>> lens;
				for (auto& w : tgm.getVocabs())
				{
					lens.emplace_back(w, tgm.arcLengthOfWord(w));
				}
				sort(lens.begin(), lens.end(), [](auto a, auto b)
				{
					return a.second < b.second;
				});
				size_t i = 0;
				for (auto& p : lens)
				{
					cout << p.first << "\t" << p.second << endl;
					if (++i >= 20) break;
				}
				cout << "==== Longest Arc Length ====" << endl;
				for (auto it = lens.end() - 20; it != lens.end(); ++it)
				{
					cout << it->first << "\t" << it->second << endl;
				}
				cout << endl;
			}
			else
			{
				string w = line.substr(1);
				cout << "==== Arc Length of " << w << " ====" << endl;
				cout << tgm.arcLengthOfWord(w) << endl << endl;
			}
		}
		else if (line[0] == '`') // estimate time of text
		{
			istringstream iss{ line.substr(1) };
			istream_iterator<string> wBegin{ iss }, wEnd{};
			vector<string> words{ wBegin, wEnd };
			auto evaluator = tgm.evaluateSent(words, args.window, 1);
			for (size_t i = 0; i <= args.initStep; ++i)
			{
				float z = i / (float)args.initStep;
				cout << tgm.unnormalizedTimePoint(z) << ": " << evaluator(z) << endl;
			}
		}
		else if (line[0] == '!') // estimate time of word
		{
			stringstream iss{ line.substr(1) };
			istream_iterator<string> wBegin{ iss }, wEnd{};
			vector<string> words{ wBegin, wEnd };
			for (size_t i = 0; i <= args.initStep; ++i)
			{
				float z = i / (float)args.initStep;
				cout << tgm.unnormalizedTimePoint(z) << ": " << 
					tgm.getWordProbByTime(words[0], tgm.unnormalizedTimePoint(z)) << endl;
			}
		}
		else // find most similar word
		{
			vector<pair<string, float>> positives, negatives;
			pair<string, float>* lastInput = nullptr;
			istringstream iss{ line };
			istream_iterator<string> wBegin{ iss }, wEnd{};
			float searchingTimePoint = tgm.getMinPoint();
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
					float t = stof(word.substr(1));
					if (t < tgm.getMinPoint() || t > tgm.getMaxPoint())
					{
						cout << "Out of time range [" << tgm.getMinPoint() << ", " << tgm.getMaxPoint() << "]" << endl;
					}

					if (lastInput)
					{
						lastInput->second = t;
						lastInput = nullptr;
					}
					else
					{
						searchingTimePoint = t;
					}
				}
				else
				{
					(sign ? negatives : positives).emplace_back(word, tgm.getMinPoint());
					lastInput = &(sign ? negatives : positives).back();
					sign = false;
				}
			}

			for (auto& p : positives)
			{
				cout << "P(@" << p.second << "|" << p.first << ") = " << tgm.getWordProbByTime(p.first, p.second) << endl;
			}

			cout << "==== Most Similar at " << searchingTimePoint << " ====" << endl;
			for (auto& p : tgm.mostSimilar(positives, negatives, searchingTimePoint, 20))
			{
				cout << get<0>(p) << '\t' << get<1>(p) << endl;
			}
			cout << endl;
		}
	}
    return 0;
}

