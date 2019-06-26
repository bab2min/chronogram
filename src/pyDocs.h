#pragma once
#include <Python.h>

#define DOC_SIGNATURE_EN(name, signature, en) PyDoc_STRVAR(name, signature "\n--\n\n" en)
#ifdef DOC_KO
#define DOC_SIGNATURE_EN_KO(name, signature, en, ko) PyDoc_STRVAR(name, signature "\n--\n\n" ko)
#else
#define DOC_SIGNATURE_EN_KO(name, signature, en, ko) PyDoc_STRVAR(name, signature "\n--\n\n" en)
#endif

DOC_SIGNATURE_EN(CGM___init____doc__,
	"Chronogram(m=100, l=6, subsampling=1e-4, word_ns=5, time_ns=5, eta=1, zeta=0.1, lambda=0.1, seed=?)",
	u8R""()"");


DOC_SIGNATURE_EN(CGM_load__doc__, 
	"load(filename)",
	u8R""()"");

