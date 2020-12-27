#pragma once
#include <type_traits>
#include <vector>
#include <unordered_map>
#include <tuple>
#include <set>
#include <limits>
#include <exception>
#include <string>
#include <iostream>
#include <cstring>

#ifdef _DEBUG
#undef _DEBUG
#include <Python.h>
#define _DEBUG
#else
#include <Python.h>
#endif

#include <frameobject.h>
#ifdef MAIN_MODULE
#else
#define NO_IMPORT_ARRAY
#endif
#define PY_ARRAY_UNIQUE_SYMBOL CHRONOGRAM_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

namespace py
{
	struct UniqueObj
	{
		PyObject* obj;
		explicit UniqueObj(PyObject* _obj = nullptr) : obj(_obj) {}
		~UniqueObj()
		{
			Py_XDECREF(obj);
		}

		UniqueObj(const UniqueObj&) = delete;
		UniqueObj& operator=(const UniqueObj&) = delete;

		UniqueObj(UniqueObj&& o) noexcept
		{
			std::swap(obj, o.obj);
		}

		UniqueObj& operator=(UniqueObj&& o) noexcept
		{
			std::swap(obj, o.obj);
			return *this;
		}

		PyObject* get() const
		{
			return obj;
		}

		PyObject* release()
		{
			auto o = obj;
			obj = nullptr;
			return o;
		}

		operator bool() const
		{
			return !!obj;
		}

		operator PyObject* () const
		{
			return obj;
		}
	};


	template<typename _Ty, typename = void>
	struct ValueBuilder;

	template<typename _Ty>
	inline PyObject* buildPyValue(_Ty&& v)
	{
		return ValueBuilder<
			typename std::remove_const<typename std::remove_reference<_Ty>::type>::type
		>{}(std::forward<_Ty>(v));
	}

	template<typename _Ty, typename _FailMsg>
	inline _Ty toCpp(PyObject* obj, _FailMsg&& fail)
	{
		if (!obj) throw std::runtime_error{ std::forward<_FailMsg>(fail) };
		return ValueBuilder<_Ty>{}._toCpp(obj, std::forward<_FailMsg>(fail));
	}

	template<typename _Ty>
	inline _Ty toCpp(PyObject* obj)
	{
		if (!obj) throw std::runtime_error{ "cannot convert null pointer into C++ type" };
		return ValueBuilder<_Ty>{}._toCpp(obj, "cannot convert Python value into C++ type");
	}

	template<typename _Ty>
	struct ValueBuilder<_Ty,
		typename std::enable_if<std::is_integral<_Ty>::value>::type>
	{
		PyObject* operator()(_Ty v)
		{
			return PyLong_FromLongLong(v);
		}

		template<typename _FailMsg>
		_Ty _toCpp(PyObject* obj, _FailMsg&&)
		{
			long long v = PyLong_AsLongLong(obj);
			if (v == -1 && PyErr_Occurred()) throw std::bad_exception{};
			return (_Ty)v;
		}
	};

	template<typename _Ty>
	struct ValueBuilder<_Ty,
		typename std::enable_if<std::is_enum<_Ty>::value>::type>
	{
		PyObject* operator()(_Ty v)
		{
			return PyLong_FromLongLong((long long)v);
		}

		template<typename _FailMsg>
		_Ty _toCpp(PyObject* obj, _FailMsg&&)
		{
			long long v = PyLong_AsLongLong(obj);
			if (v == -1 && PyErr_Occurred()) throw std::bad_exception{};
			return (_Ty)v;
		}
	};

	template<typename _Ty>
	struct ValueBuilder<_Ty,
		typename std::enable_if<std::is_floating_point<_Ty>::value>::type>
	{
		PyObject* operator()(_Ty v)
		{
			return PyFloat_FromDouble(v);
		}

		template<typename _FailMsg>
		_Ty _toCpp(PyObject* obj, _FailMsg&&)
		{
			double v = PyFloat_AsDouble(obj);
			if (v == -1 && PyErr_Occurred()) throw std::bad_exception{};
			return (_Ty)v;
		}
	};

	template<>
	struct ValueBuilder<std::string>
	{
		PyObject* operator()(const std::string& v)
		{
			return PyUnicode_FromStringAndSize(v.data(), v.size());
		}

		template<typename _FailMsg>
		std::string _toCpp(PyObject* obj, _FailMsg&&)
		{
			const char* str = PyUnicode_AsUTF8(obj);
			if (!str) throw std::bad_exception{};
			return str;
		}
	};

	template<>
	struct ValueBuilder<const char*>
	{
		PyObject* operator()(const char* v)
		{
			return PyUnicode_FromString(v);
		}

		template<typename _FailMsg>
		const char* _toCpp(PyObject* obj, _FailMsg&&)
		{
			const char* p = PyUnicode_AsUTF8(obj);
			if (!p) throw std::bad_exception{};
			return p;
		}
	};

	template<>
	struct ValueBuilder<bool>
	{
		PyObject* operator()(bool v)
		{
			return PyBool_FromLong(v);
		}

		template<typename _FailMsg>
		bool _toCpp(PyObject* obj, _FailMsg&&)
		{
			return !!PyObject_IsTrue(obj);
		}
	};

	template<>
	struct ValueBuilder<PyObject*>
	{
		PyObject* operator()(PyObject* v)
		{
			if (v)
			{
				Py_INCREF(v);
				return v;
			}
			else
			{
				Py_INCREF(Py_None);
				return Py_None;
			}
		}

		template<typename _FailMsg>
		PyObject* _toCpp(PyObject* obj, _FailMsg&&)
		{
			return obj;
		}
	};

	template<>
	struct ValueBuilder<UniqueObj>
	{
		PyObject* operator()(UniqueObj&& v)
		{
			if (v)
			{
				Py_INCREF(v);
				return v;
			}
			else
			{
				Py_INCREF(Py_None);
				return Py_None;
			}
		}
	};

	template<typename _Ty1, typename _Ty2>
	struct ValueBuilder<std::pair<_Ty1, _Ty2>>
	{
		PyObject* operator()(const std::pair<_Ty1, _Ty2>& v)
		{
			PyObject* ret = PyTuple_New(2);
			size_t id = 0;
			PyTuple_SetItem(ret, id++, buildPyValue(std::get<0>(v)));
			PyTuple_SetItem(ret, id++, buildPyValue(std::get<1>(v)));
			return ret;
		}

		template<typename _FailMsg>
		std::pair<_Ty1, _Ty2> _toCpp(PyObject* obj, _FailMsg&&)
		{
			if (PyTuple_Size(obj) != 2) throw std::runtime_error{ "input is not tuple with len=2" };
			return std::make_tuple(
				toCpp<_Ty1>(PyTuple_GetItem(obj, 0)),
				toCpp<_Ty2>(PyTuple_GetItem(obj, 1))
			);
		}
	};

	template<typename _Ty1, typename _Ty2>
	struct ValueBuilder<std::unordered_map<_Ty1, _Ty2>>
	{
		PyObject* operator()(const std::unordered_map<_Ty1, _Ty2>& v)
		{
			PyObject* ret = PyDict_New();
			for (auto& p : v)
			{
				py::UniqueObj key{ buildPyValue(p.first) }, val{ buildPyValue(p.second) };
				if (PyDict_SetItem(ret, key, val)) return nullptr;
			}
			return ret;
		}

		template<typename _FailMsg>
		std::unordered_map<_Ty1, _Ty2> _toCpp(PyObject* obj, _FailMsg&& failMsg)
		{
			std::unordered_map<_Ty1, _Ty2> ret;
			PyObject* key, * value;
			Py_ssize_t pos = 0;
			while (PyDict_Next(obj, &pos, &key, &value)) {
				ret.emplace(toCpp<_Ty1>(key), toCpp<_Ty2>(value));
			}
			if (PyErr_Occurred()) throw std::runtime_error{ failMsg };
			return ret;
		}
	};

	namespace detail
	{
		template<typename _TyTuple, size_t _Order = 0>
		typename std::enable_if< _Order >=
			std::tuple_size<typename std::remove_cv<typename std::remove_reference<_TyTuple>::type>::type>::value
		>::type setTuple(PyObject* t, _TyTuple&& v)
		{
		}

		template<typename _TyTuple, size_t _Order = 0>
		typename std::enable_if < _Order <
			std::tuple_size<typename std::remove_cv<typename std::remove_reference<_TyTuple>::type>::type>::value
		>::type setTuple(PyObject* t, _TyTuple&& v)
		{
			PyTuple_SetItem(t, _Order, buildPyValue(std::get<_Order>(v)));
			return setTuple<_TyTuple, _Order + 1>(t, std::forward<_TyTuple>(v));
		}
	}

	template<typename ..._Tys>
	struct ValueBuilder<std::tuple<_Tys...>>
	{
		PyObject* operator()(const std::tuple<_Tys...>& v)
		{
			PyObject* ret = PyTuple_New(sizeof...(_Tys));
			detail::setTuple(ret, v);
			return ret;
		}

		/*template<typename _FailMsg>
		std::tuple<_Tys...> _toCpp(PyObject* obj, _FailMsg&&)
		{
			if (PyTuple_Size(obj) != sizeof...(_Tys)) throw std::runtime_error{ "input is not tuple with len=" + std::to_string(sizeof...(_Tys)) };
			return std::make_tuple(
				toCpp<_Ty1>(PyTuple_GetItem(obj, 0)),
				toCpp<_Ty2>(PyTuple_GetItem(obj, 1))
			);
		}*/
	};

	namespace detail
	{
		template<typename _Ty>
		struct NpyType
		{
			enum {
				npy_type = -1,
			};
		};

		template<>
		struct NpyType<int8_t>
		{
			enum {
				type = NPY_INT8,
				signed_type = type,
				npy_type = type,
			};
		};

		template<>
		struct NpyType<uint8_t>
		{
			enum {
				type = NPY_UINT8,
				signed_type = NPY_INT8,
				npy_type = type,
			};
		};

		template<>
		struct NpyType<int16_t>
		{
			enum {
				type = NPY_INT16,
				signed_type = type,
				npy_type = type,
			};
		};

		template<>
		struct NpyType<uint16_t>
		{
			enum {
				type = NPY_UINT16,
				signed_type = NPY_INT16,
				npy_type = type,
			};
		};

		template<>
		struct NpyType<int32_t>
		{
			enum {
				type = NPY_INT32,
				signed_type = type,
				npy_type = type,
			};
		};

		template<>
		struct NpyType<uint32_t>
		{
			enum {
				type = NPY_UINT32,
				signed_type = NPY_INT32,
				npy_type = type,
			};
		};

		template<>
		struct NpyType<int64_t>
		{
			enum {
				type = NPY_INT64,
				signed_type = type,
				npy_type = type,
			};
		};

		template<>
		struct NpyType<uint64_t>
		{
			enum {
				type = NPY_UINT64,
				signed_type = NPY_INT64,
				npy_type = type,
			};
		};

		template<>
		struct NpyType<float>
		{
			enum {
				type = NPY_FLOAT,
				signed_type = type,
				npy_type = type,
			};
		};

		template<>
		struct NpyType<double>
		{
			enum {
				type = NPY_DOUBLE,
				signed_type = type,
				npy_type = type,
			};
		};
	}

	struct cast_to_signed_t
	{
	};

	static constexpr cast_to_signed_t cast_to_signed{};

	template<typename _Ty>
	struct ValueBuilder<std::vector<_Ty>,
		typename std::enable_if<std::is_arithmetic<_Ty>::value>::type>
	{
		PyObject* operator()(const std::vector<_Ty>& v)
		{
			npy_intp size = v.size();
			PyObject* obj = PyArray_EMPTY(1, &size, detail::NpyType<_Ty>::type, 0);
			std::memcpy(PyArray_DATA((PyArrayObject*)obj), v.data(), sizeof(_Ty) * size);
			return obj;
		}

		template<typename _FailMsg>
		std::vector<_Ty> _toCpp(PyObject* obj, _FailMsg&& failMsg)
		{
			if (detail::NpyType<_Ty>::npy_type >= 0 && PyArray_Check(obj) && PyArray_TYPE((PyArrayObject*)obj) == detail::NpyType<_Ty>::npy_type)
			{
				_Ty* ptr = (_Ty*)PyArray_GETPTR1((PyArrayObject*)obj, 0);
				return std::vector<_Ty>{ ptr, ptr + PyArray_Size(obj) };
			}
			else
			{
				UniqueObj iter{ PyObject_GetIter(obj) }, item;
				if (!iter) throw std::runtime_error{ failMsg };
				std::vector<_Ty> v;
				while ((item = UniqueObj{ PyIter_Next(iter) }))
				{
					v.emplace_back(toCpp<_Ty>(item));
				}
				if (PyErr_Occurred())
				{
					throw std::bad_exception{};
				}
				return v;
			}
		}
	};

	template<typename _Ty>
	struct ValueBuilder<std::vector<_Ty>,
		typename std::enable_if<!std::is_arithmetic<_Ty>::value>::type>
	{
		PyObject* operator()(const std::vector<_Ty>& v)
		{
			PyObject* ret = PyList_New(v.size());
			size_t id = 0;
			for (auto& e : v)
			{
				PyList_SetItem(ret, id++, buildPyValue(e));
			}
			return ret;
		}

		template<typename _FailMsg>
		std::vector<_Ty> _toCpp(PyObject* obj, _FailMsg&& failMsg)
		{
			UniqueObj iter{ PyObject_GetIter(obj) }, item;
			if (!iter) throw std::runtime_error{ failMsg };
			std::vector<_Ty> v;
			while ((item = UniqueObj{ PyIter_Next(iter) }))
			{
				v.emplace_back(toCpp<_Ty>(item));
			}
			if (PyErr_Occurred())
			{
				throw std::bad_exception{};
			}
			return v;
		}
	};

	template<typename T, typename Out, typename Msg>
	inline void transform(PyObject* iterable, Out out, Msg&& failMsg)
	{
		if (!iterable) throw std::runtime_error{ failMsg };
		UniqueObj iter{ PyObject_GetIter(iterable) }, item;
		if (!iter) throw std::runtime_error{ failMsg };
		while ((item = UniqueObj{ PyIter_Next(iter) }))
		{
			*out++ = toCpp<T>(item);
		}
		if (PyErr_Occurred())
		{
			throw std::bad_exception{};
		}
	}

	template<typename T, typename Fn, typename Msg>
	inline void foreach(PyObject* iterable, Fn&& fn, Msg&& failMsg)
	{
		if (!iterable) throw std::runtime_error{ failMsg };
		UniqueObj iter{ PyObject_GetIter(iterable) }, item;
		if (!iter) throw std::runtime_error{ failMsg };
		while ((item = UniqueObj{ PyIter_Next(iter) }))
		{
			fn(toCpp<T>(item));
		}
		if (PyErr_Occurred())
		{
			throw std::bad_exception{};
		}
	}

	template<typename _Ty>
	inline typename std::enable_if<std::is_arithmetic<_Ty>::value, PyObject*>::type
		buildPyValue(const std::vector<_Ty>& v, cast_to_signed_t)
	{
		npy_intp size = v.size();
		PyObject* obj = PyArray_EMPTY(1, &size, detail::NpyType<_Ty>::signed_type, 0);
		std::memcpy(PyArray_DATA((PyArrayObject*)obj), v.data(), sizeof(_Ty) * size);
		return obj;
	}

	template<typename _Ty>
	inline typename std::enable_if<
		!std::is_arithmetic<typename std::iterator_traits<_Ty>::value_type>::value,
		PyObject*
	>::type buildPyValue(_Ty first, _Ty last)
	{
		PyObject* ret = PyList_New(std::distance(first, last));
		size_t id = 0;
		for (; first != last; ++first)
		{
			PyList_SetItem(ret, id++, buildPyValue(*first));
		}
		return ret;
	}

	template<typename _Ty, typename _Tx>
	inline typename std::enable_if<
		!std::is_arithmetic<
		typename std::result_of<_Tx(typename std::iterator_traits<_Ty>::value_type)>::type
		>::value,
		PyObject*
	>::type buildPyValueTransform(_Ty first, _Ty last, _Tx tx)
	{
		PyObject* ret = PyList_New(std::distance(first, last));
		size_t id = 0;
		for (; first != last; ++first)
		{
			PyList_SetItem(ret, id++, buildPyValue(tx(*first)));
		}
		return ret;
	}


	template<typename _Ty>
	inline typename std::enable_if<
		std::is_arithmetic<typename std::iterator_traits<_Ty>::value_type>::value,
		PyObject*
	>::type buildPyValue(_Ty first, _Ty last)
	{
		using value_type = typename std::iterator_traits<_Ty>::value_type;
		npy_intp size = std::distance(first, last);
		PyObject* ret = PyArray_EMPTY(1, &size, detail::NpyType<value_type>::type, 0);
		size_t id = 0;
		for (; first != last; ++first, ++id)
		{
			*(value_type*)PyArray_GETPTR1((PyArrayObject*)ret, id) = *first;
		}
		return ret;
	}

	template<typename _Ty, typename _Tx>
	inline typename std::enable_if<
		std::is_arithmetic<
		typename std::result_of<_Tx(typename std::iterator_traits<_Ty>::value_type)>::type
		>::value,
		PyObject*
	>::type buildPyValueTransform(_Ty first, _Ty last, _Tx tx)
	{
		using value_type = typename std::iterator_traits<_Ty>::value_type;
		npy_intp size = std::distance(first, last);
		PyObject* ret = PyArray_EMPTY(1, &size, detail::NpyType<value_type>::type, 0);
		size_t id = 0;
		for (; first != last; ++first, ++id)
		{
			*(value_type*)PyArray_GETPTR1((PyArrayObject*)ret, id) = tx(*first);
		}
		return ret;
	}

	namespace detail
	{
		inline void setDictItem(PyObject* dict, const char** keys)
		{

		}

		template<typename _Ty, typename... _Rest>
		inline void setDictItem(PyObject* dict, const char** keys, _Ty&& value, _Rest&& ... rest)
		{
			{
				UniqueObj v{ buildPyValue(std::forward<_Ty>(value)) };
				PyDict_SetItemString(dict, keys[0], v);
			}
			return setDictItem(dict, keys + 1, std::forward<_Rest>(rest)...);
		}

		template<typename _Ty>
		inline bool isNull(_Ty v)
		{
			return false;
		}

		template<>
		inline bool isNull<PyObject*>(PyObject* v)
		{
			return !v;
		}

		inline void setDictItemSkipNull(PyObject* dict, const char** keys)
		{

		}

		template<typename _Ty, typename... _Rest>
		inline void setDictItemSkipNull(PyObject* dict, const char** keys, _Ty&& value, _Rest&& ... rest)
		{
			if (!isNull(value))
			{
				UniqueObj v{ buildPyValue(value) };
				PyDict_SetItemString(dict, keys[0], v);
			}
			return setDictItemSkipNull(dict, keys + 1, std::forward<_Rest>(rest)...);
		}

		template<size_t _n>
		inline void setTupleItem(PyObject* tuple)
		{
		}

		template<size_t _n, typename _Ty, typename... _Rest>
		inline void setTupleItem(PyObject* tuple, _Ty&& first, _Rest&&... rest)
		{
			PyTuple_SET_ITEM(tuple, _n, buildPyValue(std::forward<_Ty>(first)));
			return setTupleItem<_n + 1>(tuple, std::forward<_Rest>(rest)...);
		}
	}

	template<typename... _Rest>
	inline PyObject* buildPyDict(const char** keys, _Rest&&... rest)
	{
		PyObject* dict = PyDict_New();
		detail::setDictItem(dict, keys, std::forward<_Rest>(rest)...);
		return dict;
	}

	template<typename... _Rest>
	inline PyObject* buildPyDictSkipNull(const char** keys, _Rest&&... rest)
	{
		PyObject* dict = PyDict_New();
		detail::setDictItemSkipNull(dict, keys, std::forward<_Rest>(rest)...);
		return dict;
	}

	template<typename _Ty>
	inline void setPyDictItem(PyObject* dict, const char* key, _Ty&& value)
	{
		UniqueObj v{ buildPyValue(value) };
		PyDict_SetItemString(dict, key, v);
	}

	template<typename... _Rest>
	inline PyObject* buildPyTuple(_Rest&&... rest)
	{
		PyObject* tuple = PyTuple_New(sizeof...(_Rest));
		detail::setTupleItem<0>(tuple, std::forward<_Rest>(rest)...);
		return tuple;
	}
}
