"""
Python package `chronogram` provides types and functions for Chrono-gram Word Embedding Model.
It is written in C++ for speed and provides Python extension.

"""

isa = ''
"""
Indicate which SIMD instruction set is used for acceleration.
It can be one of `'avx2'`, `'avx'`, `'sse2'` and `'none'`.
"""

# This code is an autocomplete-hint for IDE.
# The object imported here will be overwritten by _load() function.
from _chronogram import *

def _load():
    import importlib, os
    from cpuinfo import get_cpu_info
    flags = get_cpu_info()['flags']
    env_setting = os.environ.get('CHRONOGRAM_ISA', '').split(',')
    if not env_setting[0]: env_setting = []
    isas = ['avx2', 'avx', 'sse2', 'none']
    isas = [isa for isa in isas if (env_setting and isa in env_setting) or (not env_setting and (isa in flags or isa == 'none'))]
    if not isas: raise RuntimeError("No isa option for " + str(env_setting))
    for isa in isas:
        try:
            mod_name = '_chronogram' + ('_' + isa if isa != 'none' else '')
            globals().update({k:v for k, v in vars(importlib.import_module(mod_name)).items() if not k.startswith('_')})
            return
        except:
            if isa == isas[-1]: raise
_load()
del _load
