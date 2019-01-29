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
	string load, save, eval, result, llSave, fixed, loadPrior;
	vector<string> input;
	int worker = 0, window = 4, dimension = 100;
	int order = 5, epoch = 1, negative = 5;
	int batch = 10000, minCnt = 10;
	int report = 100000;
	int nsQ = 8, initStep = 8;
	float eta = 1.f, zeta = .5f, lambda = .1f, padding = -1;
	float timeNegative = 1.f, fixedInit = 0, threshold = 0.0025f;
	float timePrior = 0;
	bool compressed = true;
	bool semEval = false;

	string evalShift, shiftMetric;
	float mixed = 0, timeA = INFINITY, timeB = INFINITY;
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

		options.add_options()
			("i,input", "Input File", cxxopts::value<string>(), "Input file path that contains documents per line")
			("l,load", "Load Model File", cxxopts::value<string>(), "Model file path to be loaded")
			("v,save", "Save Model File", cxxopts::value<string>(), "Model file path to be saved")
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
			("F,fixedInit", "Fixed Initializing Weight", cxxopts::value<float>())
			("b,batch", "Batch Docs Size", cxxopts::value<int>())
			("t,minCnt", "Min Count Threshold of Word", cxxopts::value<int>())
			("report", "", cxxopts::value<int>())
			("eta", "", cxxopts::value<float>())
			("z,zeta", "", cxxopts::value<float>())
			("lambda", "", cxxopts::value<float>())
			("p,padding", "", cxxopts::value<float>())

			("compressed", "Save as compressed", cxxopts::value<int>(), "default = 1")
			("semEval", "Print SemEval2015 Task7 Result", cxxopts::value<int>()->implicit_value("1"))
			;

		options.add_options("Text Dating")
			("eval", "Evaluation set File", cxxopts::value<string>(), "Evaluation set file path")
			("result", "Evaluation Result File", cxxopts::value<string>(), "Evaluation result file path")
			("llSave", "Save log likelihoods of evaluation", cxxopts::value<string>())
			("nsQ", "", cxxopts::value<int>())
			("initStep", "", cxxopts::value<int>())
			("threshold", "", cxxopts::value<float>())
			("timePrior", "", cxxopts::value<float>())
			("loadPrior", "", cxxopts::value<string>())
			;

		options.add_options("Semantic Shift")
			("evalShift", "File to be evaluate semantic shift", cxxopts::value<string>())
			("mixed", "mixing factor of U and V", cxxopts::value<float>())
			("timeA", "", cxxopts::value<float>())
			("timeB", "", cxxopts::value<float>())
			("shiftMetric", "", cxxopts::value<string>())
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
			READ_OPT(loadPrior, string);
			READ_OPT(llSave, string);

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
			
			READ_OPT(compressed, int);
			READ_OPT(semEval, int);

			READ_OPT(eta, float);
			READ_OPT(zeta, float);
			READ_OPT(lambda, float);
			READ_OPT(padding, float);
			READ_OPT(timeNegative, float);
			READ_OPT(fixedInit, float);
			READ_OPT(threshold, float);
			READ_OPT(timePrior, float);

			READ_OPT(evalShift, string);
			READ_OPT(timeA, float);
			READ_OPT(timeB, float);
			READ_OPT(mixed, float);
			READ_OPT(shiftMetric, string);
			
			if (args.load.empty() && args.input.empty())
			{
				throw cxxopts::OptionException("'input' or 'load' should be specified.");
			}
		}
		catch (const cxxopts::OptionException& e)
		{
			cout << "error parsing options: " << e.what() << endl;
			cout << options.help({ "", "Text Dating", "Semantic Shift" }) << endl;
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
		cout << "Padding: " << tgm.getPadding() << "\tTime Negative Weight: " << args.timeNegative << "\tFixed Initializing Weight: " << args.fixedInit << endl;

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
		tgm.train(bind(&MultipleReader::operator(), &reader, placeholders::_1),
			args.worker, args.window, args.fixedInit,
			.025f, args.batch, args.epoch, args.report);

		cout << "Finished in " << timer.getElapsed() << " sec" << endl;
		if (!args.save.empty())
		{
			cout << "Saving Model: " << args.save << endl;
			ofstream ofs{ args.save, ios_base::binary };
			tgm.saveModel(ofs, args.compressed);
		}
	}

	if (!args.evalShift.empty())
	{
		if (!isfinite(args.timeA)) args.timeA = tgm.getMinPoint();
		if (!isfinite(args.timeB)) args.timeB = tgm.getMaxPoint();

		ifstream ifs{ args.evalShift };
		string line;
		cout << "== Calculating semantic shift : " << args.evalShift 
			<< " between " << args.timeA << " and " << args.timeB << " ==" << endl;
		while (getline(ifs, line))
		{
			auto v = tgm.getEmbedding(line);
			auto u1 = tgm.getEmbedding(line, args.timeA), u2 = tgm.getEmbedding(line, args.timeB);
			float s;
			if (args.shiftMetric == "l2")
			{
				s = (u1 - u2).norm();
			}
			else
			{
				s = (v * args.mixed + u1 * (1 - args.mixed)).normalized().dot(
					(v * args.mixed + u2 * (1 - args.mixed)).normalized());
			}
			cout << line << '\t' << s << endl;
		}
		cout << endl;
	}

	map<float, float> priorMap;
	function<float(float)> priorFunc = [&priorMap](float x)->float
	{
		auto it = priorMap.lower_bound(x);
		if (it == priorMap.end()) return -INFINITY;
		if (it->first == x) return it->second;
		if (it == priorMap.begin()) return -INFINITY;
		float nx = it->first, ny = it->second;
		--it;
		float px = it->first, py = it->second;
		return py + (x - px) * (ny - py) / (nx - px);
	};

	if (!args.loadPrior.empty())
	{
		ifstream lpf{ args.loadPrior };
		string line;
		while (getline(lpf, line))
		{
			istringstream iss{ line };
			float x = INFINITY, y = INFINITY;
			iss >> x >> y;
			if (!isfinite(x) || !isfinite(y)) continue;
			priorMap.emplace(x, y);
		}
	}
	if (priorMap.empty()) priorFunc = function<float(float)>{};

	if (!args.eval.empty())
	{
		cout << "Evaluating Time Prediction: " << args.eval << endl;
		cout << "TimePriorWeight: " << args.timePrior << endl;
		Timer timer;
		ifstream ifs{ args.eval };
		if (args.result.empty()) args.result = args.eval + ".result";
		ofstream ofs{ args.result };
		ofstream ollfs;
		if (!args.llSave.empty())
		{
			ollfs = ofstream{ args.llSave };
			for (size_t i = 0; i <= args.initStep; ++i)
			{
				if (i) ollfs << '\t';
				ollfs << tgm.getMinPoint() + (tgm.getMaxPoint() - tgm.getMinPoint()) * i / args.initStep;
			}
			ollfs << endl;
		}

		float avgErr = 0, meanErr = 0;
		double score[3] = { 0, };
		size_t correct[3] = { 0, };
		size_t n = 0;
		MultipleReader reader{ {args.eval} };
		tgm.evaluate(bind(&MultipleReader::operator(), &reader, placeholders::_1), 
			[&](ChronoGramModel::EvalResult r)
		{
			ofs << r.trueTime << "\t" << r.estimatedTime << "\t" << r.ll
				<< "\t" << r.llPerWord << "\t" << r.normalizedErr << endl;
			avgErr += pow(r.normalizedErr, 2);
			meanErr += abs(r.normalizedErr);
			if (abs(r.normalizedErr) >= 0.5)
			{
				cout << r.trueTime << "\t" << r.estimatedTime << "\t" << r.normalizedErr << "\t";
				for (auto& w : r.words) cout << w << ' ';
				cout << endl;
			}

			if (ollfs)
			{
				size_t n = 0;
				for (auto ll : r.lls)
				{
					if (n++) ollfs << '\t';
					ollfs << ll;
				}
				ollfs << endl;
			}

			if (args.semEval)
			{
				static size_t span[3] = { 6, 12, 20 };
				static double offsetScore[] = { 1, 0.9, 0.85, 0.8, 0.6, 0.5, 0.4, 0.2, 0.1, 0.01 };
				float err = r.trueTime - r.estimatedTime;
				for (size_t i = 0; i < 3; ++i)
				{
					auto s = span[i];
					size_t offset = 0;
					if (abs(err) <= s / 2)
					{
						++correct[i];
					}
					else
					{
						offset = (size_t)ceil((abs(err) - s / 2) / s);
					}
					score[i] += offsetScore[min(offset, (size_t)9)];
				}
			}
			n++;
		}, args.worker, args.window, args.nsQ, priorFunc, args.timePrior, args.initStep, args.threshold);
		
		avgErr /= n;
		meanErr /= n;

		const auto printResult = [&](auto& os)
		{
			os << "Running Time: " << timer.getElapsed() << "s" << endl;
			os << "Total Docs: " << n << endl;
			os << "Avg Squared Error: " << avgErr << endl;
			os << "Avg Error: " << sqrt(avgErr) << endl;
			os << "MAE: " << meanErr << endl;

			if (args.semEval)
			{
				os << "Correct (F, M, C): " << correct[0] << ", " << correct[1] << ", " << correct[2] << endl;
				os << "Precision (F, M, C): " << correct[0] / (double)n << ", " << correct[1] / (double)n << ", " << correct[2] / (double)n << endl;
				os << "Avg Score (F, M, C): " << score[0] / n << ", " << score[1] / n << ", " << score[2] / n << endl;
			}
		};

		cout << "== Evaluating Result ==" << endl;
		printResult(cout);
		printResult(ofs);
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

				cout << "==== Shortest Angle ====" << endl;
				lens.clear();
				for (auto& w : tgm.getVocabs())
				{
					lens.emplace_back(w, tgm.angleOfWord(w));
				}
				sort(lens.begin(), lens.end(), [](auto a, auto b)
				{
					return a.second < b.second;
				});
				i = 0;
				for (auto& p : lens)
				{
					cout << p.first << "\t" << p.second / EIGEN_PI * 180 << endl;
					if (++i >= 20) break;
				}
				cout << "==== Longest Angle ====" << endl;
				for (auto it = lens.end() - 20; it != lens.end(); ++it)
				{
					cout << it->first << "\t" << it->second / EIGEN_PI * 180 << endl;
				}
				cout << endl;
			}
			else
			{
				string w = line.substr(1);
				cout << "Arc Length of " << w << " : " << tgm.arcLengthOfWord(w) << endl;
				cout << "Angle of " << w << " : " << tgm.angleOfWord(w) / EIGEN_PI * 180 << endl;
			}
		}
		else if (line[0] == '`') // estimate time of text
		{
			istringstream iss{ line.substr(1) };
			istream_iterator<string> wBegin{ iss }, wEnd{};
			vector<string> words{ wBegin, wEnd };
			auto evaluator = tgm.evaluateSent(words, args.window, args.nsQ, priorFunc, args.timePrior);
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
				cout << "Total count of " << words[0] << " : " << tgm.getWordCount(words[0]) << endl;
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
			vector<pair<string, float>> words;
			pair<string, float>* lastInput = nullptr;
			stringstream iss{ line.substr(1) };
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
					words.emplace_back(word, -INFINITY);
					lastInput = &words.back();
				}
			}
			for(auto& w : words)
			{
				auto vec = isfinite(w.second) 
					? tgm.getEmbedding(w.first, w.second)
					: tgm.getEmbedding(w.first);
				printf("[");
				for (size_t i = 0; i < vec.size(); ++i)
				{
					printf("%.3f, ", vec(i));
				}
				printf("]\n");
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
		else if (line[0] == '%') // find most similar word using mixed vector
		{
			istringstream iss{ line.substr(1) };
			float mixed = 0;
			iss >> mixed;
			vector<pair<string, float>> positives, negatives;
			pair<string, float>* lastInput = nullptr;
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
					(sign ? negatives : positives).emplace_back(word, searchingTimePoint);
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
			for (auto& p : tgm.mostSimilar(positives, negatives, searchingTimePoint, mixed, 20))
			{
				if (get<1>(p) <= 0) break;
				cout << left << setw(12) << get<0>(p) << '\t' << get<1>(p) << '\t' << get<2>(p) << endl;
			}
			cout << endl;
		}
		else if (line[0] == '^') // calculate shift of words
		{
			istringstream iss{ line.substr(1) };
			size_t minCnt = 100, show = 40;
			float time1 = 0, time2 = 0, m = 0;
			iss >> minCnt >> time1 >> time2 >> m >> show;
			auto ret = tgm.calcShift(minCnt, time1, time2, m);
			cout << "== Shift between " << time1 << " and " << time2 << " ==" << endl;
			for (size_t i = 0; i < 10 && i < ret.size(); ++i)
			{
				cout << ret[i].first << "\t" << ret[i].second << endl;
			}
			cout << "  ...  " << endl;
			for (size_t i = max(ret.size(), (size_t)show) - show; i < ret.size(); ++i)
			{
				cout << ret[i].first << "\t" << ret[i].second << endl;
			}
			cout << endl;
		}
		else if (line[0] == '&')
		{
			istringstream iss{ line.substr(1) };
			float m = 0, time = 0;
			string src, t;
			vector<string> targets;
			iss >> m >> time >> src;
			while (iss >> t) targets.emplace_back(t);
			cout << "== Sum of similarities from " << src << " : " << tgm.sumSimilarity(src, targets, time, m) << endl;
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
					(sign ? negatives : positives).emplace_back(word, searchingTimePoint);
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
			for (auto& p : tgm.mostSimilar(positives, negatives, searchingTimePoint, 0, 20))
			{
				if (get<1>(p) <= 0) break;
				cout << left << setw(12) << get<0>(p) << '\t' << get<1>(p) << '\t' << get<2>(p) << endl;
			}
			cout << endl;

			cout << "==== Most Similar Overall ====" << endl;
			for (auto& p : tgm.mostSimilar(positivesO, negativesO, 20))
			{
				if (get<1>(p) <= 0) break;
				cout << left << setw(12) << get<0>(p) << '\t' << get<1>(p) << endl;
			}
			cout << endl;
		}
	}
    return 0;
}

