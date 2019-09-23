#pragma once

#include <iostream>
#include <vector>
#include <map>

class imstream
{
private:
	const char* ptr, *begin, *end;
public:
	imstream(const char* _ptr, size_t len) : ptr(_ptr), begin(_ptr), end(_ptr + len)
	{
	}

	template<class _Ty>
	const _Ty& read()
	{
		if (end - ptr < sizeof(_Ty)) throw std::ios_base::failure(std::string{ "reading type '" } +typeid(_Ty).name() + "' failed");
		auto p = (_Ty*)ptr;
		ptr += sizeof(_Ty);
		return *p;
	}

	bool read(void* dest, size_t size)
	{
		if (end - ptr < size) return false;
		std::memcpy(dest, ptr, size);
		ptr += size;
		return true;
	}

	void exceptions(int)
	{
		// dummy functions
	}

	const char* get() const
	{
		return ptr;
	}

	size_t tellg() const
	{
		return ptr - begin;
	}

	bool seekg(std::streamoff distance, std::ios_base::seek_dir dir = std::ios_base::beg)
	{
		if (dir == std::ios_base::beg)
		{
			if (distance < 0 || distance > end - begin) return false;
			ptr = begin + distance;
		}
		else if (dir == std::ios_base::cur)
		{
			if (ptr + distance < begin || ptr + distance > end) return false;
			ptr += distance;
		}
		else if (dir == std::ios_base::end)
		{
			if (distance > 0 || end + distance < begin) return false;
			ptr = end + distance;
		}
		else return false;
		return true;
	}
};

template<class _Ty> inline void writeToBinStream(std::ostream& os, const _Ty& v);
template<class _Ty, class _Istream> inline _Ty readFromBinStream(_Istream& is);
template<class _Ty, class _Istream> inline void readFromBinStream(_Istream& is, _Ty& v);

template<class _Ty>
inline typename std::enable_if<std::is_fundamental<_Ty>::value || std::is_enum<_Ty>::value>::type writeToBinStreamImpl(std::ostream& os, const _Ty& v)
{
	if (!os.write((const char*)&v, sizeof(_Ty))) throw std::ios_base::failure(std::string{ "writing type '" } +typeid(_Ty).name() + "' failed");
}

template<class _Ty, class _Istream>
inline typename std::enable_if<std::is_fundamental<_Ty>::value || std::is_enum<_Ty>::value>::type readFromBinStreamImpl(_Istream& is, _Ty& v)
{
	if (!is.read((char*)&v, sizeof(_Ty))) throw std::ios_base::failure(std::string{ "reading type '" } +typeid(_Ty).name() + "' failed");
}

template<typename _Elem>
inline void writeToBinStreamImpl(std::ostream& os, const std::basic_string<_Elem>& v)
{
	writeToBinStream<uint32_t>(os, v.size());
	if (!os.write((const char*)&v[0], v.size() * sizeof(_Elem))) throw std::ios_base::failure(std::string{ "writing type '" } +typeid(std::basic_string<_Elem>).name() + "' failed");
}

template<typename _Elem, class _Istream>
inline void readFromBinStreamImpl(_Istream& is, std::basic_string<_Elem>& v)
{
	v.resize(readFromBinStream<uint32_t>(is));
	if (!is.read((char*)&v[0], v.size() * sizeof(_Elem))) throw std::ios_base::failure(std::string{ "reading type '" } +typeid(std::basic_string<_Elem>).name() + "' failed");
}

template<class _Ty1, class _Ty2>
inline void writeToBinStreamImpl(std::ostream& os, const typename std::pair<_Ty1, _Ty2>& v)
{
	writeToBinStream(os, v.first);
	writeToBinStream(os, v.second);
}

template<class _Ty1, class _Ty2, class _Istream>
inline void readFromBinStreamImpl(_Istream& is, typename std::pair<_Ty1, _Ty2>& v)
{
	v.first = readFromBinStream<_Ty1>(is);
	v.second = readFromBinStream<_Ty2>(is);
}


template<class _Ty1, class _Ty2>
inline void writeToBinStreamImpl(std::ostream& os, const typename std::map<_Ty1, _Ty2>& v)
{
	writeToBinStream<uint32_t>(os, v.size());
	for (auto& p : v)
	{
		writeToBinStream(os, p);
	}
}

template<class _Ty1, class _Ty2, class _Istream>
inline void readFromBinStreamImpl(_Istream& is, typename std::map<_Ty1, _Ty2>& v)
{
	size_t len = readFromBinStream<uint32_t>(is);
	v.clear();
	for (size_t i = 0; i < len; ++i)
	{
		v.emplace(readFromBinStream<std::pair<_Ty1, _Ty2>>(is));
	}
}


template<class _Ty1>
inline void writeToBinStreamImpl(std::ostream& os, const typename std::vector<_Ty1>& v)
{
	writeToBinStream<uint32_t>(os, v.size());
	for (auto& p : v)
	{
		writeToBinStream(os, p);
	}
}

template<class _Ty1, class _Istream>
inline void readFromBinStreamImpl(_Istream& is, typename std::vector<_Ty1>& v)
{
	size_t len = readFromBinStream<uint32_t>(is);
	v.clear();
	for (size_t i = 0; i < len; ++i)
	{
		v.emplace_back(readFromBinStream<_Ty1>(is));
	}
}

template<typename _Ty1, int _Rows, int _Cols>
inline void writeToBinStreamImpl(std::ostream& os, const typename Eigen::Matrix<_Ty1, _Rows, _Cols>& v)
{
	for (size_t i = 0; i < v.size(); ++i)
	{
		writeToBinStream(os, v.data()[i]);
	}
}

template<typename _Ty1, int _Rows, int _Cols, class _Istream>
inline void readFromBinStreamImpl(_Istream& is, typename Eigen::Matrix<_Ty1, _Rows, _Cols>& v)
{
	for (size_t i = 0; i < v.size(); ++i)
	{
		readFromBinStream(is, v.data()[i]);
	}
}

void writeFloatVL(std::ostream& os, float f);
float readFloatVL(std::istream& is);
float readFloatVL(imstream& is);

template<typename _Ty1, int _Rows, int _Cols>
inline void writeToBinStreamCompressed(std::ostream& os, const typename Eigen::Matrix<_Ty1, _Rows, _Cols>& v)
{
	for (size_t i = 0; i < v.size(); ++i)
	{
		writeFloatVL(os, v.data()[i]);
	}
}

template<typename _Ty1, int _Rows, int _Cols, class _Istream>
inline void readFromBinStreamCompressed(_Istream& is, typename Eigen::Matrix<_Ty1, _Rows, _Cols>& v)
{
	for (size_t i = 0; i < v.size(); ++i)
	{
		v.data()[i] = readFloatVL(is);
	}
}

template<class _Ty>
inline void writeToBinStream(std::ostream& os, const _Ty& v)
{
	writeToBinStreamImpl(os, v);
}


template<class _Ty, class _Istream>
inline _Ty readFromBinStream(_Istream& is)
{
	_Ty v;
	readFromBinStreamImpl(is, v);
	return v;
}

template<class _Ty, class _Istream>
inline void readFromBinStream(_Istream& is, _Ty& v)
{
	readFromBinStreamImpl(is, v);
}
