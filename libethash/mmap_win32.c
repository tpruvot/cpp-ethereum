/* mmap() replacement for Windows
 *
 * Author: Mike Frysinger <vapier@gentoo.org>
 * Placed into the public domain
 */

/* References:
 * CreateFileMapping: http://msdn.microsoft.com/en-us/library/aa366537(VS.85).aspx
 * CloseHandle:       http://msdn.microsoft.com/en-us/library/ms724211(VS.85).aspx
 * MapViewOfFile:     http://msdn.microsoft.com/en-us/library/aa366761(VS.85).aspx
 * UnmapViewOfFile:   http://msdn.microsoft.com/en-us/library/aa366882(VS.85).aspx
 */

#include <io.h>
#include <windows.h>
#include "FileAPI.h"
#include "mmap.h"

#ifdef __USE_FILE_OFFSET64
# define DWORD_HI(x) (x >> 32)
# define DWORD_LO(x) ((x) & 0xffffffff)
#else
# define DWORD_HI(x) (0)
# define DWORD_LO(x) (x)
#endif

extern char* ethash_io_lastdag_filename();
void* map_chunks[2] = { NULL, NULL };

void mmap_get_chuncks(void* *chunks)
{
	chunks[0] = map_chunks[0];
	chunks[1] = map_chunks[1]; // offset = chunk[0] + 0x3000.0000 - 1024
}

void* mmap(void* start, size_t len, int prot, int flags, int fd, off_t offset)
{
	if (prot & ~(PROT_READ | PROT_WRITE | PROT_EXEC))
		return MAP_FAILED;
	if (fd == -1) {
		if (!(flags & MAP_ANON) || offset)
			return MAP_FAILED;
	} else if (flags & MAP_ANON)
		return MAP_FAILED;

	DWORD flProtect;
	if (prot & PROT_WRITE) {
		if (prot & PROT_EXEC)
			flProtect = PAGE_EXECUTE_READWRITE;
		else
			flProtect = PAGE_READWRITE;
	} else if (prot & PROT_EXEC) {
		if (prot & PROT_READ)
			flProtect = PAGE_EXECUTE_READ;
		else if (prot & PROT_EXEC)
			flProtect = PAGE_EXECUTE;
	} else
		flProtect = PAGE_READONLY;

	off_t end = len + offset;
	HANDLE mmap_fd, h;
	if (fd == -1)
		mmap_fd = INVALID_HANDLE_VALUE;
	else
		mmap_fd = (HANDLE)_get_osfhandle(fd);
#if 0
	_close(fd);

	char *fn = ethash_io_lastdag_filename();
	mmap_fd = CreateFile(fn,
		GENERIC_READ, FILE_SHARE_READ,
		NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, NULL);
	flProtect = PAGE_READONLY;
	prot &= ~PROT_WRITE;
#endif

	//flProtect |= SEC_COMMIT | SEC_LARGE_PAGES;

	h = CreateFileMapping(mmap_fd, NULL, flProtect, DWORD_HI(end), DWORD_LO(end), NULL);
	if (h == NULL)
		return MAP_FAILED;

	DWORD dwDesiredAccess;
	if (prot & PROT_WRITE)
		dwDesiredAccess = FILE_MAP_WRITE;
	else
		dwDesiredAccess = FILE_MAP_READ;
	if (prot & PROT_EXEC)
		dwDesiredAccess |= FILE_MAP_EXECUTE;
	if (flags & MAP_PRIVATE)
		dwDesiredAccess |= FILE_MAP_COPY;
	void *ret = MapViewOfFile(h, dwDesiredAccess, DWORD_HI(offset), DWORD_LO(offset), 0);
	if (ret == NULL) {
		// seems to be the max for x86 binaries without large pages
		ret = MapViewOfFile(h, dwDesiredAccess, DWORD_HI(offset), DWORD_LO(offset), MAP_CHUNK_MAX_SIZE);
		if (ret != NULL) {
			map_chunks[0] = ret;
			offset += MAP_CHUNK_MAX_SIZE - MAP_CHUNK_EXTRAPAD; // some extra to be sure we have full nodes, offset require alignment...
			ret = MapViewOfFile(h, dwDesiredAccess, DWORD_HI(offset), DWORD_LO(offset), (len - MAP_CHUNK_MAX_SIZE) + MAP_CHUNK_EXTRAPAD);
			if (ret != NULL) {
				map_chunks[1] = ret;
				ret = MAP_HALF;
			}
		}
	}
	if (ret == NULL) {
		ret = MAP_FAILED;
	}
	// since we are handling the file ourselves with fd, close the Windows Handle here
	CloseHandle(h);
	return ret;
}

void munmap(void* addr, size_t len)
{
	UnmapViewOfFile(addr);
}

#undef DWORD_HI
#undef DWORD_LO
