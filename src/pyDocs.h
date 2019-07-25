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

DOC_SIGNATURE_EN(CGM_save__doc__,
	"save(filename, compressed=True)",
	u8R""()"");

DOC_SIGNATURE_EN(CGM_build_vocab__doc__,
	"build_vocab(reader, min_cnt=10, workers=0)",
	u8R""()"");

DOC_SIGNATURE_EN(CGM_initialize__doc__,
	"initialize(reader, workers=0, window_len=4, start_lr=0.025, end_lr=0.00025, batch_size=1000, epochs=1, report=10000)",
	u8R""()"");

DOC_SIGNATURE_EN(CGM_train__doc__,
	"train(reader, workers=0, window_len=4, start_lr=0.025, end_lr=0.00025, batch_size=1000, epochs=1, report=10000)",
	u8R""()"");

DOC_SIGNATURE_EN(CGM_build_vocab_gn__doc__,
	"build_vocab_gn(vocab_file, min_time, max_time)",
	u8R""()"");

DOC_SIGNATURE_EN(CGM_initialize_gn__doc__,
	"initialize_gn(ngram_file, max_items=-1, workers=0, start_lr=0.025, end_lr=0.00025, batch_size=1000, epochs=1, report=10000)",
	u8R""()"");

DOC_SIGNATURE_EN(CGM_train_gn__doc__,
	"train_gn(ngram_file, max_items=-1, workers=0, start_lr=0.025, end_lr=0.00025, batch_size=1000, epochs=1, report=10000)",
	u8R""()"");

DOC_SIGNATURE_EN(CGM_most_similar__doc__,
	"most_similar(positives, negatives=None, time=None, m=0, topN=10)",
	u8R""()"");

DOC_SIGNATURE_EN(CGM_similarity__doc__,
	"similarity(word1, time1, word2, time2)",
	u8R""()"");

DOC_SIGNATURE_EN(CGM_embedding__doc__,
	"embedding(word, time)",
	u8R""()"");

DOC_SIGNATURE_EN(CGM_most_similar_s__doc__,
	"most_similar_s(positives, negatives=None, topN=10)",
	u8R""()"");

DOC_SIGNATURE_EN(CGM_similarity_s__doc__,
	"similarity_s(word1, word2)",
	u8R""()"");

DOC_SIGNATURE_EN(CGM_embedding_s__doc__,
	"embedding_s(word)",
	u8R""()"");

DOC_SIGNATURE_EN(CGM_p_time__doc__,
	"p_time(time)",
	u8R""()"");

DOC_SIGNATURE_EN(CGM_p_time_word__doc__,
	"p_time_word(time, word)",
	u8R""()"");

DOC_SIGNATURE_EN(CGM_evaluator__doc__,
	"evaluator(words, window_len=4, ns_q=16, time_prior_fun=None, time_prior_weight=0)",
	u8R""()"");

DOC_SIGNATURE_EN(CGE___init____doc__,
	"",
	u8R""()"");

DOC_SIGNATURE_EN(CGV___init____doc__,
	"",
	u8R""()"");