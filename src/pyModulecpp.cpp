#include <fstream>
#include <iostream>

#include "ChronoGramModel.h"
#include "PyUtils.h"

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
static PyObject* CGM_save(CGMObject*, PyObject* args, PyObject *kwargs);
static PyObject* CGM_buildVocab(CGMObject*, PyObject* args, PyObject *kwargs);
static PyObject* CGM_train(CGMObject*, PyObject* args, PyObject *kwargs);
static PyObject* CGM_mostSimilar(CGMObject*, PyObject* args, PyObject *kwargs);
static PyObject* CGM_similarity(CGMObject*, PyObject* args, PyObject *kwargs);
static PyObject* CGM_getEmbedding(CGMObject*, PyObject* args, PyObject *kwargs);

static PyMethodDef CGM_methods[] =
{
	{ "load", (PyCFunction)CGM_load, METH_STATIC | METH_VARARGS | METH_KEYWORDS, CGM_load__doc__ },
	{ "save", (PyCFunction)CGM_save, METH_VARARGS | METH_KEYWORDS, CGM_save__doc__ },
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
