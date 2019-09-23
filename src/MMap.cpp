#include "MMap.h"

using namespace std;

#ifdef _WIN32
MMap::MMap(const string & filepath)
{
	hFile = CreateFileA(filepath.c_str(), GENERIC_READ, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_READONLY, nullptr);
	if (hFile == INVALID_HANDLE_VALUE) throw ios_base::failure("Cannot open '" + filepath + "'");
	hFileMap = CreateFileMapping(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
	if (hFileMap == nullptr) throw ios_base::failure("Cannot open '" + filepath + "' Code:" + to_string(GetLastError()));
	view = (const char*)MapViewOfFile(hFileMap, FILE_MAP_READ, 0, 0, 0);
	DWORD high;
	len = GetFileSize(hFile, &high);
	len |= (size_t)high << 32;
}

MMap::~MMap()
{
	UnmapViewOfFile(view);
	hFileMap.~HandleGuard();
}
#else

#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>

MMap::MMap(const string & filepath)
{
	fd = open(filepath.c_str(), O_RDONLY);
	if (fd == -1) throw ios_base::failure("Cannot open '" + filepath + "'");
	struct stat sb;
	if (fstat(fd, &sb) < 0) throw ios_base::failure("Cannot open '" + filepath + "'");
	len = sb.st_size;
	view = (const char*)mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
	if(view == MAP_FAILED) throw ios_base::failure("Mapping failed");
}

MMap::~MMap()
{
	munmap((void*)view, len);
}
#endif