#pragma once
#include <string>
#include <iostream>

#ifdef _WIN32
#include <Windows.h>
class HandleGuard
{
	HANDLE handle;
public:
	HandleGuard(HANDLE _handle = nullptr) : handle(_handle)
	{
	}

	HandleGuard(const HandleGuard&) = delete;
	HandleGuard& operator =(const HandleGuard&) = delete;
	
	HandleGuard(HandleGuard&& o)
	{
		std::swap(handle, o.handle);
	}

	HandleGuard& operator=(HandleGuard&& o)
	{
		std::swap(handle, o.handle);
		return *this;
	}

	~HandleGuard()
	{
		if (handle && handle != INVALID_HANDLE_VALUE)
		{
			CloseHandle(handle);
			handle = nullptr;
		}
	}

	operator HANDLE() const
	{
		return handle;
	}
};

class MMap
{
	const char* view = nullptr;
	size_t len = 0;
	HandleGuard hFile, hFileMap;
public:
	MMap(const std::string& filepath);

	MMap(const MMap&) = delete;
	MMap& operator=(const MMap&) = delete;

	MMap(MMap&&) = default;
	MMap& operator=(MMap&&) = default;

	~MMap();

	const char* get() const { return view; }
	size_t size() const { return len; }
};
#else
#include <unistd.h>
class FDGuard
{
	int fd;
public:
	FDGuard(int _fd = 0) : fd(_fd)
	{
	}

	FDGuard(const FDGuard&) = delete;
	FDGuard& operator =(const FDGuard&) = delete;

	FDGuard(FDGuard&& o)
	{
		std::swap(fd, o.fd);
	}

	FDGuard& operator=(FDGuard&& o)
	{
		std::swap(fd, o.fd);
		return *this;
	}

	~FDGuard()
	{
		if (fd && fd != -1)
		{
			close(fd);
			fd = 0;
		}
	}

	operator int() const
	{
		return fd;
	}
};

class MMap
{
	const char* view = nullptr;
	size_t len = 0;
	FDGuard fd;
public:
	MMap(const std::string& filepath);

	MMap(const MMap&) = delete;
	MMap& operator=(const MMap&) = delete;

	MMap(MMap&&) = default;
	MMap& operator=(MMap&&) = default;

	~MMap();

	const char* get() const { return view; }
	size_t size() const { return len; }
};
#endif