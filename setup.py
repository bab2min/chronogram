from setuptools import setup, Extension
from codecs import open
import os, os.path, struct, platform
from setuptools.command.install import install

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

sources = []
for f in os.listdir(os.path.join(here, 'src')):
    if f.endswith('.cpp') and not f.endswith('main.cpp'): sources.append('src/' + f)

if platform.system() == 'Windows': 
    cargs = ['/O2', '/MT', '/Gy']
    arch_levels = {'':'', 'sse2':'/arch:SSE2', 'avx':'/arch:AVX', 'avx2':'/arch:AVX2'}
elif platform.system() == 'Darwin': 
    cargs = ['-std=c++1y', '-O3', '-fpermissive', '-stdlib=libc++']
    arch_levels = {'':'-march=native'}
else:
    cargs = ['-std=c++1y', '-O3', '-fpermissive']
    arch_levels = {'':'-march=native'}

if struct.calcsize('P') < 8: arch_levels = {k:v for k, v in arch_levels.items() if k in ('', 'sse2')}
else: arch_levels = {k:v for k, v in arch_levels.items() if k not in ('sse2',)}

modules = []
for arch, aopt in arch_levels.items():
    module_name = '_chronogram' + ('_' + arch if arch else '')
    modules.append(Extension(module_name,
                    libraries = [],
                    include_dirs=['include'],
                    sources = sources,
                    define_macros=[('MODULE_NAME', 'PyInit_' + module_name)],
                    extra_compile_args=cargs + ([aopt] if aopt else [])))


setup(
    name='chronogram',

    version='0.1.3',

    description='Chrono-gram, the diachronic word embedding model based on Word2vec Skip-gram with Chebyshev approximation',
    long_description=long_description,

    url='https://github.com/bab2min/chronogram',

    author='bab2min',
    author_email='bab2min@gmail.com',

    license='MIT License',

    classifiers=[
        'Development Status :: 3 - Alpha',

        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries",
        "Topic :: Text Processing :: Linguistic",
		"Topic :: Scientific/Engineering :: Information Analysis",

        "License :: OSI Approved :: MIT License",

        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
		"Operating System :: Microsoft :: Windows :: Windows Vista",
		"Operating System :: Microsoft :: Windows :: Windows 7",
		"Operating System :: Microsoft :: Windows :: Windows 8",
		"Operating System :: Microsoft :: Windows :: Windows 8.1",
		"Operating System :: Microsoft :: Windows :: Windows 10",
		"Operating System :: POSIX"
    ],
    install_requires=['py-cpuinfo'],
    keywords='NLP,Word2vec,Word Embedding',

    packages = ['chronogram'],
    include_package_data=True,
    ext_modules = modules
)
