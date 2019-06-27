#include <fstream>
#include <iostream>

#include "ChronoGramModel.h"
#include "PyUtils.h"
#include "pyDocs.h"

using namespace std;

static PyObject* gModule;

struct CGMObject
{
	PyObject_HEAD;
	ChronoGramModel* inst;
	bool isPrepared;

	static void dealloc(CGMObject* self)
	{
		if (self->inst)
		{
			delete self->inst;
		}
		Py_TYPE(self)->tp_free((PyObject*)self);
	}
};

static PyObject* CGM_load(PyObject*, PyObject* args, PyObject *kwargs);
static PyObject* CGM_save(CGMObject* self, PyObject* args, PyObject *kwargs);
static PyObject* CGM_buildVocab(CGMObject* self, PyObject* args, PyObject *kwargs);
static PyObject* CGM_train(CGMObject* self, PyObject* args, PyObject *kwargs);
static PyObject* CGM_mostSimilar(CGMObject* self, PyObject* args, PyObject *kwargs);
static PyObject* CGM_similarity(CGMObject* self, PyObject* args, PyObject *kwargs);
static PyObject* CGM_getEmbedding(CGMObject* self, PyObject* args, PyObject *kwargs);

static PyMethodDef CGM_methods[] =
{
	{ "load", (PyCFunction)CGM_load, METH_STATIC | METH_VARARGS | METH_KEYWORDS, CGM_load__doc__ },
	{ "save", (PyCFunction)CGM_save, METH_VARARGS | METH_KEYWORDS, CGM_save__doc__ },
	{ "build_vocab", (PyCFunction)CGM_buildVocab, METH_VARARGS | METH_KEYWORDS, CGM_build_vocab__doc__ },
	{ "train", (PyCFunction)CGM_train, METH_VARARGS | METH_KEYWORDS, CGM_train__doc__ },
	{ "most_similar", (PyCFunction)CGM_mostSimilar, METH_VARARGS | METH_KEYWORDS, CGM_most_similar__doc__ },
	{ "similarity", (PyCFunction)CGM_similarity, METH_VARARGS | METH_KEYWORDS, CGM_similarity__doc__ },
	{ "get_embedding", (PyCFunction)CGM_getEmbedding, METH_VARARGS | METH_KEYWORDS, CGM_get_embedding__doc__ },
	{ nullptr }
};

static PyGetSetDef CGM_getseters[] = {
	{ nullptr },
};


static int CGM_init(CGMObject *self, PyObject *args, PyObject *kwargs)
{
	size_t M = 100, L = 6;
	float subsampling = 1e-4;
	size_t NS = 5, TNS = 5;
	float eta = 1, zeta = 0.1f, lambda = 0.1f;
	size_t seed = std::random_device{}();
	static const char* kwlist[] = { "m", "l", "subsampling", "word_ns", "time_ns", "eta", "zeta", "lambda", "seed", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nnfnnfffn", (char**)kwlist, 
		&M, &L, &subsampling, &NS, &TNS, &eta, &zeta, &lambda, &seed)) return -1;
	try
	{
		ChronoGramModel* inst = new ChronoGramModel(M, L, subsampling, NS, TNS, eta, zeta, lambda, seed);
		self->inst = inst;
		self->isPrepared = false;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return -1;
	}
	return 0;
}

static PyTypeObject CGM_type = {
	PyVarObject_HEAD_INIT(NULL, 0)
	"chronogram.Chronogram",             /* tp_name */
	sizeof(CGMObject), /* tp_basicsize */
	0,                         /* tp_itemsize */
	(destructor)CGMObject::dealloc, /* tp_dealloc */
	0,                         /* tp_print */
	0,                         /* tp_getattr */
	0,                         /* tp_setattr */
	0,                         /* tp_reserved */
	0,                         /* tp_repr */
	0,                         /* tp_as_number */
	0,                         /* tp_as_sequence */
	0,                         /* tp_as_mapping */
	0,                         /* tp_hash  */
	0,                         /* tp_call */
	0,                         /* tp_str */
	0,                         /* tp_getattro */
	0,                         /* tp_setattro */
	0,                         /* tp_as_buffer */
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
	CGM___init____doc__,           /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	0,                         /* tp_iter */
	0,                         /* tp_iternext */
	CGM_methods,             /* tp_methods */
	0,						 /* tp_members */
	CGM_getseters,                         /* tp_getset */
	0,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)CGM_init,      /* tp_init */
	PyType_GenericAlloc,
	PyType_GenericNew,
};


PyObject * CGM_load(PyObject *, PyObject * args, PyObject * kwargs)
{
	const char* filename;
	static const char* kwlist[] = { "filename", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s", (char**)kwlist, &filename)) return nullptr;
	try
	{
		auto* p = (CGMObject*)PyObject_CallObject((PyObject*)&CGM_type, Py_BuildValue("()"));
		try
		{
			ifstream is{ filename };
			*p->inst = ChronoGramModel::loadModel(is);
		}
		catch (const exception&)
		{
			Py_XDECREF(p);
			throw runtime_error{ "wrong model file '" + string{filename} + "'" };
		}
		return (PyObject*)p;
	}
	catch (const bad_exception&)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

PyObject * CGM_save(CGMObject * self, PyObject * args, PyObject * kwargs)
{
	const char* filename;
	int compressed = 1;
	static const char* kwlist[] = { "filename", "compressed", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|n", (char**)kwlist, &filename, &compressed)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		ofstream os{ filename };
		self->inst->saveModel(os, !!compressed);
		Py_INCREF(Py_None);
		return Py_None;
	}
	catch (const bad_exception&)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

function<ChronoGramModel::ResultReader()> makeCGMReader(PyObject* reader)
{
	return [=]()
	{
		PyObject* r = PyObject_GetIter(reader);
		if(!r) throw runtime_error{ "'reader' must be iterable" };
		
		struct ReaderObj
		{
			PyObject* r;

			~ReaderObj()
			{
				Py_XDECREF(r);
			}

			ChronoGramModel::ReadResult operator()()
			{
				ChronoGramModel::ReadResult ret;
				auto* item = PyIter_Next(r);
				if (item)
				{
					if (PyTuple_Size(item) != 2)
					{
						auto repr = PyObject_Repr(item);
						string srepr = PyUnicode_AsUTF8(repr);
						Py_XDECREF(repr);
						throw runtime_error{ "wrong return value of 'reader' : " + srepr };
					}
					auto* wordIter = PyObject_GetIter(PyTuple_GetItem(item, 0));
					if (!wordIter) throw runtime_error{ "first item of tuple must be list of str" };
					ret.words = py::makeIterToVector(wordIter);
					Py_XDECREF(wordIter);
					ret.timePoint = PyFloat_AsDouble(PyTuple_GetItem(item, 1));
					if (ret.timePoint == -1 && PyErr_Occurred()) throw bad_exception{};
				}
				else
				{
					if (PyErr_Occurred()) throw bad_exception{};
					ret.stop = true;
				}
				return ret;
			}
		};

		auto sr = make_shared<ReaderObj>(r);
		return [sr]() { return sr->operator()(); };
	};
}

PyObject * CGM_buildVocab(CGMObject * self, PyObject * args, PyObject * kwargs)
{
	PyObject* reader = nullptr;
	size_t minCnt = 10, workers = 0;
	static const char* kwlist[] = { "reader", "min_cnt", "workers", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|nn", (char**)kwlist, &reader, &minCnt, &workers)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		self->inst->buildVocab(makeCGMReader(reader), minCnt, workers);
		Py_INCREF(Py_None);
		return Py_None;
	}
	catch (const bad_exception&)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

PyObject * CGM_train(CGMObject * self, PyObject * args, PyObject * kwargs)
{
	PyObject* reader = nullptr;
	size_t workers = 0, windowLen = 4, batchSents = 1000, epochs = 1, report = 10000;
	float initEpochs = 0, startLR = 0.025;
	static const char* kwlist[] = { "reader", "workers", "window_len", "init_epochs", "start_lr", "batch_size", "epochs", "report", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|nnffnnn", (char**)kwlist, 
		&reader, &workers, &windowLen, &initEpochs, &startLR, &batchSents, &epochs, &report)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		self->inst->train(makeCGMReader(reader), workers, windowLen, initEpochs, startLR, batchSents, epochs, report);
		Py_INCREF(Py_None);
		return Py_None;
	}
	catch (const bad_exception&)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

PyObject * CGM_mostSimilar(CGMObject * self, PyObject * args, PyObject * kwargs)
{
	PyObject *positives, *negatives = nullptr;
	float time = -INFINITY, m = 0;
	size_t topN = 10;
	static const char* kwlist[] = { "positives", "negatives", "time", "m", "top_n", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Offn", (char**)kwlist,
		&positives, &negatives, &time, &m, &topN)) return nullptr;
	
	const auto& parseWord = [](PyObject* obj)
	{
		if (PyTuple_Size(obj) != 2) throw runtime_error{ "'positives' and 'negatives' should be (word :str, time :float) or its list" };
		const char* word = PyUnicode_AsUTF8(PyTuple_GetItem(obj, 0));
		if (!word) throw bad_exception{};
		float time = PyFloat_AsDouble(PyTuple_GetItem(obj, 1));
		if (time == -1 && PyErr_Occurred()) throw bad_exception{};
		return make_pair(string{ word }, time);
	};

	const auto& parseWords = [&](PyObject* obj)
	{
		vector<pair<string, float>> ret;
		if (PyTuple_Check(obj))
		{
			ret.emplace_back(parseWord(obj));
		}
		else
		{
			PyObject *iter = PyObject_GetIter(obj), *item;
			py::AutoReleaser arIter{ iter };
			if(!iter) throw runtime_error{ "'positives' and 'negatives' should be (word :str, time :float) or its list" };
			while (item = PyIter_Next(iter))
			{
				ret.emplace_back(parseWord(item));
			}
		}
		return ret;
	};
	
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		vector<pair<string, float>> pos, neg;
		if(time == -INFINITY) time = self->inst->getMinPoint();
		pos = parseWords(positives);
		if (negatives) neg = parseWords(negatives);
		return py::buildPyValue(self->inst->mostSimilar(pos, neg, time, m, topN));
	}
	catch (const bad_exception&)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

PyObject * CGM_similarity(CGMObject * self, PyObject * args, PyObject * kwargs)
{
	const char *word1, *word2;
	float time1, time2;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		static const char* kwlist[] = { "word1", "time1", "word2", "time2", nullptr };
		if (PyArg_ParseTupleAndKeywords(args, kwargs, "sfsf", (char**)kwlist,
			&word1, &time1, &word2, &time2))
		{
			return py::buildPyValue(self->inst->similarity(word1, time1, word2, time2));
		}

		static const char* kwlist2[] = { "word1", "word2", nullptr };
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ss", (char**)kwlist,
			&word1, &word2)) return nullptr;
		PyErr_Clear();
		return py::buildPyValue(self->inst->similarity(word1, word2));
	}
	catch (const bad_exception&)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

PyObject * CGM_getEmbedding(CGMObject * self, PyObject * args, PyObject * kwargs)
{
	return nullptr;
}


PyMODINIT_FUNC MODULE_NAME()
{
	static PyModuleDef mod =
	{
		PyModuleDef_HEAD_INIT,
		"chronogram",
		"Chrono-gram, the diachronic word embedding model for Python",
		-1,
		nullptr,
	};

	if (PyType_Ready(&CGM_type) < 0) return nullptr;

	gModule = PyModule_Create(&mod);
	if (!gModule) return nullptr;


#ifdef __AVX2__
	PyModule_AddStringConstant(gModule, "isa", "avx2");
#elif defined(__AVX__)
	PyModule_AddStringConstant(gModule, "isa", "avx");
#elif defined(__SSE2__) || defined(__x86_64__) || defined(_WIN64)
	PyModule_AddStringConstant(gModule, "isa", "sse2");
#else
	PyModule_AddStringConstant(gModule, "isa", "none");
#endif

	Py_INCREF(&CGM_type);
	PyModule_AddObject(gModule, "Chronogram", (PyObject*)&CGM_type);
	return gModule;
}
