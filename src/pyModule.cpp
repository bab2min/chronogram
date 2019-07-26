#include <fstream>
#include <iostream>
#include <memory>

#include "ChronoGramModel.h"
#include "DataReader.h"
#include "PyUtils.h"
#include "pyDocs.h"

using namespace std;

static PyObject* gModule;


#define DEFINE_GETTER(GETTER)\
static PyObject* CGM_##GETTER(CGMObject *self, void *closure)\
{\
	try\
	{\
		if (!self->inst) throw runtime_error{ "inst is null" };\
		return py::buildPyValue(self->inst->GETTER());\
	}\
	catch (const bad_exception&)\
	{\
		return nullptr;\
	}\
	catch (const exception& e)\
	{\
		PyErr_SetString(PyExc_Exception, e.what());\
		return nullptr;\
	}\
}\


struct CGMObject
{
	PyObject_HEAD;
	ChronoGramModel* inst;

	static void dealloc(CGMObject* self)
	{
		if (self->inst)
		{
			delete self->inst;
		}
		Py_TYPE(self)->tp_free((PyObject*)self);
	}

	static int init(CGMObject *self, PyObject *args, PyObject *kwargs)
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
		}
		catch (const exception& e)
		{
			PyErr_SetString(PyExc_Exception, e.what());
			return -1;
		}
		return 0;
	}

	static PyObject* getVocabs(CGMObject *self, void* closure);
};

struct CGVObject
{
	PyObject_HEAD;
	CGMObject* parentObj;

	static void dealloc(CGVObject* self)
	{
		Py_XDECREF(self->parentObj);
		Py_TYPE(self)->tp_free((PyObject*)self);
	}

	static int init(CGVObject *self, PyObject *args, PyObject *kwargs)
	{
		static const char* kwlist[] = { "parent", nullptr };
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char**)kwlist, &self->parentObj)) return -1;
		Py_INCREF(self->parentObj);
		return 0;
	}

	static Py_ssize_t length(CGVObject* self)
	{
		try
		{
			return self->parentObj->inst->getVocabs().size();
		}
		catch (const exception& e)
		{
			PyErr_SetString(PyExc_Exception, e.what());
			return -1;
		}
	}

	static PyObject* getItem(CGVObject* self, Py_ssize_t key)
	{
		try
		{
			if (key < self->parentObj->inst->getVocabs().size()) return py::buildPyValue(self->parentObj->inst->getVocabs()[key]);
			PyErr_SetString(PyExc_IndexError, "");
			return nullptr;
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
};

struct CGEObject
{
	PyObject_HEAD;
	CGMObject* parentObj;
	ChronoGramModel::LLEvaluater* inst;
	PyObject* timePrior;

	static void dealloc(CGEObject* self)
	{
		if (self->inst)
		{
			delete self->inst;
		}
		Py_XDECREF(self->timePrior);
		Py_XDECREF(self->parentObj);
		Py_TYPE(self)->tp_free((PyObject*)self);
	}

	static int init(CGEObject *self, PyObject *args, PyObject *kwargs)
	{
		static const char* kwlist[] = { "parent", "inst", "time_prior_fun", nullptr };
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OnO", (char**)kwlist,
			&self->parentObj, &self->inst, &self->timePrior)) return -1;
		Py_INCREF(self->parentObj);
		Py_INCREF(self->timePrior);
		return 0;
	}

	static PyObject* call(CGEObject *self, PyObject *args, PyObject *kwargs)
	{
		float time;
		static const char* kwlist[] = { "time", nullptr };
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "f", (char**)kwlist,
			&time)) return nullptr;
		try
		{
			time = self->parentObj->inst->normalizedTimePoint(time);
			if (time < 0 || time > 1)
			{
				char buf[256];
				snprintf(buf, 256, "'time' is out of time range [%g, %g]",
					self->parentObj->inst->getMinPoint(),
					self->parentObj->inst->getMaxPoint());
				PyErr_SetString(PyExc_Exception, buf);
				return nullptr;
			}
			return py::buildPyValue(self->inst->operator()(time));
		}
		catch (const bad_exception&)
		{
			return nullptr;
		}
	}
};

static PyObject* CGM_load(PyObject*, PyObject* args, PyObject *kwargs);
static PyObject* CGM_save(CGMObject* self, PyObject* args, PyObject *kwargs);
static PyObject* CGM_buildVocab(CGMObject* self, PyObject* args, PyObject *kwargs);
static PyObject* CGM_initialize(CGMObject* self, PyObject* args, PyObject *kwargs);
static PyObject* CGM_train(CGMObject* self, PyObject* args, PyObject *kwargs);
static PyObject* CGM_buildVocabGN(CGMObject* self, PyObject* args, PyObject *kwargs);
static PyObject* CGM_initializeGN(CGMObject* self, PyObject* args, PyObject *kwargs);
static PyObject* CGM_trainGN(CGMObject* self, PyObject* args, PyObject *kwargs);
static PyObject* CGM_mostSimilar(CGMObject* self, PyObject* args, PyObject *kwargs);
static PyObject* CGM_similarity(CGMObject* self, PyObject* args, PyObject *kwargs);
static PyObject* CGM_getEmbedding(CGMObject* self, PyObject* args, PyObject *kwargs);
static PyObject* CGM_mostSimilarStatic(CGMObject* self, PyObject* args, PyObject *kwargs);
static PyObject* CGM_similarityStatic(CGMObject* self, PyObject* args, PyObject *kwargs);
static PyObject* CGM_getEmbeddingStatic(CGMObject* self, PyObject* args, PyObject *kwargs);
static PyObject* CGM_evaluator(CGMObject* self, PyObject* args, PyObject *kwargs);
static PyObject* CGM_pTime(CGMObject* self, PyObject* args, PyObject *kwargs);
static PyObject* CGM_pTimeWord(CGMObject* self, PyObject* args, PyObject *kwargs);

static PyMethodDef CGM_methods[] =
{
	{ "load", (PyCFunction)CGM_load, METH_STATIC | METH_VARARGS | METH_KEYWORDS, CGM_load__doc__ },
	{ "save", (PyCFunction)CGM_save, METH_VARARGS | METH_KEYWORDS, CGM_save__doc__ },
	{ "build_vocab", (PyCFunction)CGM_buildVocab, METH_VARARGS | METH_KEYWORDS, CGM_build_vocab__doc__ },
	{ "initialize", (PyCFunction)CGM_initialize, METH_VARARGS | METH_KEYWORDS, CGM_initialize__doc__ },
	{ "train", (PyCFunction)CGM_train, METH_VARARGS | METH_KEYWORDS, CGM_train__doc__ },
	{ "build_vocab_gn", (PyCFunction)CGM_buildVocabGN, METH_VARARGS | METH_KEYWORDS, CGM_build_vocab_gn__doc__ },
	{ "initialize_gn", (PyCFunction)CGM_initializeGN, METH_VARARGS | METH_KEYWORDS, CGM_initialize_gn__doc__ },
	{ "train_gn", (PyCFunction)CGM_trainGN, METH_VARARGS | METH_KEYWORDS, CGM_train_gn__doc__ },
	{ "most_similar", (PyCFunction)CGM_mostSimilar, METH_VARARGS | METH_KEYWORDS, CGM_most_similar__doc__ },
	{ "similarity", (PyCFunction)CGM_similarity, METH_VARARGS | METH_KEYWORDS, CGM_similarity__doc__ },
	{ "embedding", (PyCFunction)CGM_getEmbedding, METH_VARARGS | METH_KEYWORDS, CGM_embedding__doc__ },
	{ "most_similar_s", (PyCFunction)CGM_mostSimilarStatic, METH_VARARGS | METH_KEYWORDS, CGM_most_similar_s__doc__ },
	{ "similarity_s", (PyCFunction)CGM_similarityStatic, METH_VARARGS | METH_KEYWORDS, CGM_similarity_s__doc__ },
	{ "embedding_s", (PyCFunction)CGM_getEmbeddingStatic, METH_VARARGS | METH_KEYWORDS, CGM_embedding_s__doc__ },
	{ "p_time", (PyCFunction)CGM_pTime, METH_VARARGS | METH_KEYWORDS, CGM_p_time__doc__ },
	{ "p_time_word", (PyCFunction)CGM_pTimeWord, METH_VARARGS | METH_KEYWORDS, CGM_p_time_word__doc__ },
	{ "evaluator", (PyCFunction)CGM_evaluator, METH_VARARGS | METH_KEYWORDS, CGM_evaluator__doc__ },
	{ nullptr }
};

DEFINE_GETTER(getM);
DEFINE_GETTER(getL);
DEFINE_GETTER(getZeta);
DEFINE_GETTER(getLambda);
DEFINE_GETTER(getPadding);
DEFINE_GETTER(getMinPoint);
DEFINE_GETTER(getMaxPoint);
DEFINE_GETTER(getTPBias);
DEFINE_GETTER(getTPThreshold);

static int CGM_setPadding(CGMObject *self, PyObject *value, void *closure)
{
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		float v = PyFloat_AsDouble(value);
		if (v == -1 && PyErr_Occurred()) throw bad_exception{};
		self->inst->setPadding(v);
		return 0;
	}
	catch (const bad_exception&)
	{
		return -1;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return -1;
	}
}

static int CGM_setTPBias(CGMObject *self, PyObject *value, void *closure) 
{
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" }; 
		float v = PyFloat_AsDouble(value);
		if (v == -1 && PyErr_Occurred()) throw bad_exception{};
		self->inst->setTPBias(v);
		return 0;
	}
	catch (const bad_exception&)
	{
		return -1; 
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what()); 
		return -1; 
	}
}

static int CGM_setTPThreshold(CGMObject *self, PyObject *value, void *closure)
{
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		float v = PyFloat_AsDouble(value);
		if (v == -1 && PyErr_Occurred()) throw bad_exception{};
		self->inst->setTPThreshold(v);
		return 0;
	}
	catch (const bad_exception&)
	{
		return -1;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return -1;
	}
}

static PyGetSetDef CGM_getseters[] = {
	{ (char*)"m", (getter)CGM_getM, nullptr, (char*)"embedding dimension", NULL },
	{ (char*)"l", (getter)CGM_getL, nullptr, (char*)"chebyshev approximation order", NULL },
	{ (char*)"zeta", (getter)CGM_getZeta, nullptr, (char*)"zeta, mixing factor", NULL },
	{ (char*)"lambda_v", (getter)CGM_getLambda, nullptr, (char*)"lambda", NULL },
	{ (char*)"min_time", (getter)CGM_getMinPoint, nullptr, (char*)"time range", NULL },
	{ (char*)"max_time", (getter)CGM_getMaxPoint, nullptr, (char*)"time range", NULL },
	{ (char*)"vocabs", (getter)CGMObject::getVocabs, nullptr, (char*)"vocabularies in the model", NULL },
	{ (char*)"padding", (getter)CGM_getPadding, (setter)CGM_setPadding, (char*)"padding", NULL },
	{ (char*)"tp_bias", (getter)CGM_getTPBias, (setter)CGM_setTPBias, (char*)"bias of whole temporal distribution", NULL },
	{ (char*)"tp_threshold", (getter)CGM_getTPThreshold, (setter)CGM_setTPThreshold, (char*)"filtering threshold on temporal probability", NULL },
	{ nullptr },
};


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
	(initproc)CGMObject::init,      /* tp_init */
	PyType_GenericAlloc,
	PyType_GenericNew,
};

static PySequenceMethods CGV_seqs = {
	(lenfunc)CGVObject::length,
	0,
	0,
	(ssizeargfunc)CGVObject::getItem,
};

static PyTypeObject CGV_type = {
	PyVarObject_HEAD_INIT(NULL, 0)
	"chronogram._Vocabs",             /* tp_name */
	sizeof(CGVObject), /* tp_basicsize */
	0,                         /* tp_itemsize */
	(destructor)CGVObject::dealloc, /* tp_dealloc */
	0,                         /* tp_print */
	0,                         /* tp_getattr */
	0,                         /* tp_setattr */
	0,                         /* tp_reserved */
	0,                         /* tp_repr */
	0,                         /* tp_as_number */
	&CGV_seqs,                         /* tp_as_sequence */
	0,                         /* tp_as_mapping */
	0,                         /* tp_hash  */
	0,  /* tp_call */
	0,                         /* tp_str */
	0,                         /* tp_getattro */
	0,                         /* tp_setattro */
	0,                         /* tp_as_buffer */
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
	CGV___init____doc__,           /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	0,                         /* tp_iter */
	0,                         /* tp_iternext */
	0,             /* tp_methods */
	0,						 /* tp_members */
	0,                         /* tp_getset */
	0,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)CGVObject::init,      /* tp_init */
	PyType_GenericAlloc,
	PyType_GenericNew,
};


static PyTypeObject CGE_type = {
	PyVarObject_HEAD_INIT(NULL, 0)
	"chronogram._LLEvaluator",             /* tp_name */
	sizeof(CGEObject), /* tp_basicsize */
	0,                         /* tp_itemsize */
	(destructor)CGEObject::dealloc, /* tp_dealloc */
	0,                         /* tp_print */
	0,                         /* tp_getattr */
	0,                         /* tp_setattr */
	0,                         /* tp_reserved */
	0,                         /* tp_repr */
	0,                         /* tp_as_number */
	0,                         /* tp_as_sequence */
	0,                         /* tp_as_mapping */
	0,                         /* tp_hash  */
	(ternaryfunc)CGEObject::call,  /* tp_call */
	0,                         /* tp_str */
	0,                         /* tp_getattro */
	0,                         /* tp_setattro */
	0,                         /* tp_as_buffer */
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
	CGE___init____doc__,           /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	0,                         /* tp_iter */
	0,                         /* tp_iternext */
	0,             /* tp_methods */
	0,						 /* tp_members */
	0,                         /* tp_getset */
	0,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)CGEObject::init,      /* tp_init */
	PyType_GenericAlloc,
	PyType_GenericNew,
};

PyObject* CGMObject::getVocabs(CGMObject *self, void* closure)
{
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		return PyObject_CallObject((PyObject*)&CGV_type, Py_BuildValue("(N)", self));
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
			ifstream is{ filename, ios_base::binary };
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
		ofstream os{ filename, ios_base::binary };
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

			ReaderObj(PyObject* _r) : r(_r) {}

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

PyObject * CGM_initialize(CGMObject * self, PyObject * args, PyObject * kwargs)
{
	PyObject* reader = nullptr;
	size_t workers = 0, windowLen = 4, batchSents = 1000, report = 10000;
	float startLR = 0.025, endLR = 0.00025, epochs = 1;
	static const char* kwlist[] = { "reader", "workers", "window_len", "start_lr", "end_lr", "batch_size", "epochs", "report", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|nnffnfn", (char**)kwlist,
		&reader, &workers, &windowLen, &startLR, &endLR, &batchSents, &epochs, &report)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		self->inst->template train<true>(makeCGMReader(reader), workers, windowLen, startLR, endLR, batchSents, epochs, report);
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
	size_t workers = 0, windowLen = 4, batchSents = 1000, report = 10000;
	float startLR = 0.025, endLR = 0.00025, epochs = 1;
	static const char* kwlist[] = { "reader", "workers", "window_len", "start_lr", "end_lr", "batch_size", "epochs", "report", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|nnffnfn", (char**)kwlist, 
		&reader, &workers, &windowLen, &startLR, &endLR, &batchSents, &epochs, &report)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		self->inst->train(makeCGMReader(reader), workers, windowLen, startLR, endLR, batchSents, epochs, report);
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

PyObject * CGM_buildVocabGN(CGMObject * self, PyObject * args, PyObject * kwargs)
{
	const char* vocabFile = nullptr;
	float minT, maxT;
	static const char* kwlist[] = { "vocab_file", "min_time", "max_time", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sff", (char**)kwlist, &vocabFile, &minT, &maxT)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		ifstream vocab{ vocabFile };
		if (!vocab)
		{
			PyErr_SetString(PyExc_IOError, vocabFile);
			throw bad_exception{};
		}
		self->inst->buildVocabFromDict([&vocab]()
		{
			string line;
			getline(vocab, line);
			istringstream iss{ line };
			string w;
			getline(iss, w, '\t');
			uint64_t cnt = 0;
			iss >> cnt;
			return make_pair(w, cnt);
		}, minT, maxT);
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

PyObject * CGM_initializeGN(CGMObject * self, PyObject * args, PyObject * kwargs)
{
	const char* ngram;
	size_t workers = 0, maxItems = -1, batchSents = 1000, report = 10000;
	float startLR = 0.025, endLR = 0.00025, epochs = 1;
	static const char* kwlist[] = { "ngram_file", "max_items", "workers", "start_lr", "end_lr", "batch_size", "epochs", "report", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|nnffnfn", (char**)kwlist,
		&ngram, &maxItems, &workers, &startLR, &endLR, &batchSents, &epochs, &report)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		self->inst->template trainFromGNgram<true>(GNgramBinaryReader::factory(ngram), maxItems, workers, startLR, endLR, batchSents, epochs, report);
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

PyObject * CGM_trainGN(CGMObject * self, PyObject * args, PyObject * kwargs)
{
	const char* ngram;
	size_t workers = 0, maxItems = -1, batchSents = 1000, report = 10000;
	float startLR = 0.025, endLR = 0.00025, epochs = 1;
	static const char* kwlist[] = { "ngram_file", "max_items", "workers", "start_lr", "end_lr", "batch_size", "epochs", "report", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|nnffnfn", (char**)kwlist,
		&ngram, &maxItems, &workers, &startLR, &endLR, &batchSents, &epochs, &report)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		self->inst->trainFromGNgram(GNgramBinaryReader::factory(ngram), maxItems, workers, startLR, endLR, batchSents, epochs, report);
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
		static const char* kwlist[] = { "word1", "time1", "word2", "time2", nullptr };
		if (PyArg_ParseTupleAndKeywords(args, kwargs, "sfsf", (char**)kwlist,
			&word1, &time1, &word2, &time2)) return nullptr;

		if (!self->inst) throw runtime_error{ "inst is null" };
		return py::buildPyValue(self->inst->similarity(word1, time1, word2, time2));
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
	try
	{
		const char *word;
		float time;
		static const char* kwlist[] = { "word", "time", nullptr };
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sf", (char**)kwlist,
			&word, &time)) return nullptr;

		if (!self->inst) throw runtime_error{ "inst is null" };
		
		auto v = self->inst->getEmbedding(word, time);
		return py::buildPyValue(v.data(), v.data() + v.size());
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


PyObject * CGM_mostSimilarStatic(CGMObject * self, PyObject * args, PyObject * kwargs)
{
	PyObject *positives, *negatives = nullptr;
	size_t topN = 10;
	static const char* kwlist[] = { "positives", "negatives", "top_n", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Ofn", (char**)kwlist,
		&positives, &negatives, &topN)) return nullptr;

	const auto& parseWords = [](PyObject* obj)
	{
		vector<string> ret;
		if (PyUnicode_Check(obj))
		{
			ret.emplace_back(PyUnicode_AsUTF8(obj));
		}
		else
		{
			PyObject *iter = PyObject_GetIter(obj);
			if (!iter) throw runtime_error{ "'positives' and 'negatives' should be str or its list" };
			ret = py::makeIterToVector(iter);
			Py_XDECREF(iter);
		}
		return ret;
	};

	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		vector<string> pos, neg;
		pos = parseWords(positives);
		if (negatives) neg = parseWords(negatives);
		return py::buildPyValue(self->inst->mostSimilarStatic(pos, neg, topN));
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

PyObject * CGM_similarityStatic(CGMObject * self, PyObject * args, PyObject * kwargs)
{
	const char *word1, *word2;
	try
	{
		static const char* kwlist[] = { "word1", "word2", nullptr };
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ss", (char**)kwlist,
			&word1, &word2)) return nullptr;

		if (!self->inst) throw runtime_error{ "inst is null" };
		return py::buildPyValue(self->inst->similarityStatic(word1, word2));
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

PyObject * CGM_getEmbeddingStatic(CGMObject * self, PyObject * args, PyObject * kwargs)
{
	try
	{
		const char *word;
		static const char* kwlist[] = { "word", nullptr };
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s", (char**)kwlist,
			&word)) return nullptr;

		if (!self->inst) throw runtime_error{ "inst is null" };

		auto v = self->inst->getEmbedding(word);
		return py::buildPyValue(v.data(), v.data() + v.size());
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

PyObject * CGM_evaluator(CGMObject * self, PyObject * args, PyObject * kwargs)
{
	try
	{
		PyObject *words, *timePrior = nullptr;
		size_t windowLen = 4, nsQ = 16;
		float timePriorWeight = 0;
		static const char* kwlist[] = { "words", "window_len", "ns_q", "time_prior_fun", "time_prior_weight", nullptr };
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|nnOf", (char**)kwlist,
			&words, &windowLen, &nsQ, &timePrior, &timePriorWeight)) return nullptr;
		
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* wordIter = PyObject_GetIter(words);
		if (!wordIter) throw runtime_error{ "'words' must be list of str" };
		auto wordsVector = py::makeIterToVector(wordIter);
		Py_XDECREF(wordIter);
		auto* e = new ChronoGramModel::LLEvaluater{ self->inst->evaluateSent(wordsVector, windowLen, nsQ, [timePrior](float t)
		{
			if(!timePrior) return 0.f;
			auto* ret = PyObject_CallObject(timePrior, Py_BuildValue("(f)", t));
			if (!ret) throw bad_exception{};
			py::AutoReleaser ar{ ret };
			float v = PyFloat_AsDouble(ret);
			if (v == -1 && PyErr_Occurred()) throw bad_exception{};
			return v;
		}, timePriorWeight) };
		return PyObject_CallObject((PyObject*)&CGE_type, Py_BuildValue(timePrior ? "(NnN)" : "(Nns)", self, e, timePrior));
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

PyObject * CGM_pTime(CGMObject * self, PyObject * args, PyObject * kwargs)
{
	try
	{
		float time;
		static const char* kwlist[] = { "time", nullptr };
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "f", (char**)kwlist, &time)) return nullptr;

		if (!self->inst) throw runtime_error{ "inst is null" };
		return py::buildPyValue(self->inst->getTimePrior(time));
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

PyObject * CGM_pTimeWord(CGMObject * self, PyObject * args, PyObject * kwargs)
{
	try
	{
		float time;
		const char* word;
		static const char* kwlist[] = { "time", "word", nullptr };
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "fs", (char**)kwlist, &time, &word)) return nullptr;

		if (!self->inst) throw runtime_error{ "inst is null" };
		return py::buildPyValue(self->inst->getWordProbByTime(word, time));
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


PyMODINIT_FUNC MODULE_NAME()
{
	static PyModuleDef mod =
	{
		PyModuleDef_HEAD_INIT,
		"chronogram",
		"Chrono-gram, The Diachronic Word Embedding Model",
		-1,
		nullptr,
	};

	if (PyType_Ready(&CGM_type) < 0) return nullptr;
	if (PyType_Ready(&CGV_type) < 0) return nullptr;
	if (PyType_Ready(&CGE_type) < 0) return nullptr;

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
	Py_INCREF(&CGE_type);
	PyModule_AddObject(gModule, "_LLEvaluator", (PyObject*)&CGE_type);
	Py_INCREF(&CGV_type);
	PyModule_AddObject(gModule, "_Vocabs", (PyObject*)&CGE_type);
	return gModule;
}
