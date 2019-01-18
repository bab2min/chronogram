#include <vector>
#include <array>
#include <map>
#include <algorithm>
#include <unordered_map>
#include <numeric>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <iterator>
#include <fstream>
#include <random>
#include <thread>

#include "cxxopts.hpp"
#include "ChronoGramModel.h"

using namespace std;

struct Args
{
	string load, save, eval, result, fixed;
	vector<string> input;
	int worker = 0, window = 4, dimension = 100;
	int order = 5, epoch = 1, negative = 5;
	int batch = 10000, minCnt = 10;
	int report = 100000;
	int nsQ = 8, initStep = 32;
	float eta = 1.f, zeta = .5f, lambda = .1f, padding = -1;
	float timeNegative = 5.f;
};

struct MultipleReader
{
	vector<string> files;
	size_t currentId = 0;
	ifstream ifs;

	MultipleReader(const vector<string>& _files) : files(_files)
	{
		ifs = ifstream{ files[currentId] };
	}

	ChronoGramModel::ReadResult operator()(size_t id)
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
			("i,input", "Input File", cxxopts::value<string>(), "Input file path that contains documents per line")
			("l,load", "Load Model File", cxxopts::value<string>(), "Model file path to be loaded")
			("v,save", "Save Model File", cxxopts::value<string>(), "Model file path to be saved")
			("eval", "Evaluation set File", cxxopts::value<string>(), "Evaluation set file path")
			("result", "Evaluation Result File", cxxopts::value<string>(), "Evaluation result file path")
			("f,fixed", "Fixed Word List File", cxxopts::value<string>())
			("h,help", "Help")
			("version", "Version")

			("w,worker", "Number of Workes", cxxopts::value<int>(), "The number of workers(thread) for inferencing model, default value is 0 which means the number of cores in system")
			("W,window", "Size of Window", cxxopts::value<int>())
			("d,dimension", "Embedding Dimension", cxxopts::value<int>())
			("r,order", "Order of Chebyshev Polynomial", cxxopts::value<int>())
			("e,epoch", "Number of Epoch", cxxopts::value<int>())
			("n,negative", "Negative Sampling Size", cxxopts::value<int>())
			("T,timeNegative", "Time Negative Weight", cxxopts::value<float>())
			("b,batch", "Batch Docs Size", cxxopts::value<int>())
			("t,minCnt", "Min Count Threshold of Word", cxxopts::value<int>())
			("report", "", cxxopts::value<int>())
			("nsQ", "", cxxopts::value<int>())
			("initStep", "", cxxopts::value<int>())
			("eta", "", cxxopts::value<float>())
			("z,zeta", "", cxxopts::value<float>())
			("lambda", "", cxxopts::value<float>())
			("p,padding", "", cxxopts::value<float>())
			;

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
			READ_OPT(eval, string);
			READ_OPT(result, string);
			READ_OPT(fixed, string);

			if (result.count("input")) args.input.emplace_back(result["input"].as<string>());
			for (size_t i = 1; i < argc; ++i)
			{
				args.input.emplace_back(argv[i]);
			}

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
			READ_OPT(zeta, float);
			READ_OPT(lambda, float);
			READ_OPT(padding, float);
			READ_OPT(timeNegative, float);
			
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

	ChronoGramModel tgm{ (size_t)args.dimension, (size_t)args.order, 1e-4, (size_t)args.negative,
		args.timeNegative, args.eta, args.zeta, args.lambda };
	if (args.padding >= 0)
	{
		tgm.setPadding(args.padding);
	}

	if (!args.load.empty())
	{
		cout << "Loading Model: " << args.load << endl;
		ifstream ifs{ args.load, ios_base::binary };
		tgm = ChronoGramModel::loadModel(ifs);
		cout << "Dimension: " << tgm.getM() << "\tOrder: " << tgm.getL() << endl;
		cout << "Workers: " << (args.worker ? args.worker : thread::hardware_concurrency()) << endl;
		cout << "Zeta: " << tgm.getZeta() << "\tLambda: " << tgm.getLambda() << endl;
		cout << "Padding: " << tgm.getPadding() << endl;
	}
	else if (!args.input.empty())
	{
		cout << "Dimension: " << args.dimension << "\tOrder: " << args.order << "\tNegative Sampling: " << args.negative << endl;
		cout << "Workers: " << (args.worker ? args.worker : thread::hardware_concurrency()) << "\tBatch: " << args.batch << "\tEpochs: " << args.epoch << endl;
		cout << "Eta: " << args.eta << "\tZeta: " << args.zeta << "\tLambda: " << args.lambda << endl;
		cout << "Padding: " << tgm.getPadding() << "\tTime Negative Weight: " << args.timeNegative << endl;

		cout << "Training Input: ";
		for (auto& s : args.input)
		{
			cout << s << "  ";
		}
		cout << endl;

		Timer timer;
		MultipleReader reader{ args.input };
		tgm.buildVocab(bind(&MultipleReader::operator(), &reader, placeholders::_1), args.minCnt);
		cout << "MinCnt: " << args.minCnt << "\tVocab Size: " << tgm.getVocabs().size() 
			<< "\tTotal Words: " << tgm.getTotalWords() << endl;
		if(!args.fixed.empty())
		{
			ifstream ifs{ args.fixed };
			string line;
			size_t numFixedWords = 0;
			while (getline(ifs, line))
			{
				while (!line.empty() && isspace(line.back())) line.pop_back();
				if (tgm.addFixedWord(line)) numFixedWords++;
			}
			cout << numFixedWords << " fixed words are loaded." << endl;
		}
		tgm.train(bind(&MultipleReader::operator(), &reader, placeholders::_1), args.worker, args.window, .025f, args.batch,
			args.epoch, args.report);

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
		float avgErr = 0;
		size_t n = 0;
		MultipleReader reader{ {args.eval} };
		tgm.evaluate(bind(&MultipleReader::operator(), &reader, placeholders::_1), 
			[&](ChronoGramModel::EvalResult r)
		{
			ofs << r.trueTime << "\t" << r.estimatedTime << "\t" << r.ll
				<< "\t" << r.llPerWord << "\t" << r.normalizedErr << endl;
			avgErr += pow(r.normalizedErr, 2);
			n++;
		}, args.worker, args.window, args.nsQ, args.initStep);
		
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
		if (line[0] == '~') // calculate arc length
		{
			if (line.size() == 1) // find shortest arc length
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
				float z = tgm.getMinPoint() + (i / (float)args.initStep) * (tgm.getMaxPoint() - tgm.getMinPoint());
				cout << z << ": " << evaluator(tgm.normalizedTimePoint(z)) << endl;
			}
		}
		else if (line[0] == '!') // estimate time of word
		{
			stringstream iss{ line.substr(1) };
			istream_iterator<string> wBegin{ iss }, wEnd{};
			vector<string> words{ wBegin, wEnd };
			if (words.empty())
			{
				for (size_t i = 0; i <= args.initStep; ++i)
				{
					float z = tgm.getMinPoint() + (i / (float)args.initStep) * (tgm.getMaxPoint() - tgm.getMinPoint());
					cout << z << ": " << tgm.getTimePrior(z) << endl;
				}
			}
			else
			{
				for (size_t i = 0; i <= args.initStep; ++i)
				{
					float z = tgm.getMinPoint() + (i / (float)args.initStep) * (tgm.getMaxPoint() - tgm.getMinPoint());
					float w = tgm.getWordProbByTime(words[0], z);
					float t = tgm.getTimePrior(z);
					cout << z << ": " << w << "\t" << w / t << endl;
				}
			}
		}
		else if (line[0] == '#') // get embedding
		{
			stringstream iss{ line.substr(1) };
			istream_iterator<string> wBegin{ iss }, wEnd{};
			vector<string> words{ wBegin, wEnd };
			for (auto& w : words)
			{
				auto mat = tgm.getEmbedding(w);
				for (size_t i = 0; i < mat.cols(); ++i)
				{
					//cout << setprecision(3);
					for (size_t j = 0; j < mat.rows(); ++j)
					{
						cout << mat(j, i) << ", ";
					}
					cout << endl;
				}
			}
		}
		else if (line[0] == '$') // similarity between two word
		{
			vector<pair<string, float>> words;
			pair<string, float>* lastInput = nullptr;
			istringstream iss{ line.substr(1) };
			istream_iterator<string> wBegin{ iss }, wEnd{};
			for (; wBegin != wEnd; ++wBegin)
			{
				auto word = *wBegin;
				if (word[0] == '@')
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
				}
				else
				{
					words.emplace_back(word, tgm.getMinPoint());
					lastInput = &words.back();
				}
			}
			if (words.size() < 2)
			{
				cout << "Input two words!!" << endl;
			}
			else
			{
				cout << "Similarity between '" << words[0].first << "' @" << words[0].second
					<< " and '" << words[1].first << "' @" << words[1].second
					<< " : " << tgm.similarity(words[0].first, words[0].second, words[1].first, words[1].second) << endl;
				cout << "Overall similarity between '" << words[0].first << "' and '" << words[1].first
					<< "' : " << tgm.similarity(words[0].first,words[1].first) << endl;
			}
		}
		else // find most similar word
		{
			vector<pair<string, float>> positives, negatives;
			vector<string> positivesO, negativesO;
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
					(sign ? negativesO : positivesO).emplace_back(word);
					lastInput = &(sign ? negatives : positives).back();
					sign = false;
				}
			}

			for (auto& p : positives)
			{
				float wp = tgm.getWordProbByTime(p.first, p.second);
				float tp = tgm.getTimePrior(p.second);
				cout << "P(@" << p.second << " | " << p.first << ") = " << wp << ",\tP(@" << p.second << ") = " << tp << endl;
			}

			cout << "==== Most Similar at " << searchingTimePoint << " ====" << endl;
			for (auto& p : tgm.mostSimilar(positives, negatives, searchingTimePoint, 20))
			{
				if (get<1>(p) <= 0) break;
				cout << left << setw(12) << get<0>(p) << '\t' << get<1>(p) << '\t' << get<2>(p) << endl;
			}
			cout << endl;

			cout << "==== Most Similar Overall ====" << endl;
			for (auto& p : tgm.mostSimilar(positivesO, negativesO, 20))
			{
				cout << left << setw(12) << get<0>(p) << '\t' << get<1>(p) << endl;
			}
			cout << endl;
		}
	}
    return 0;
}

