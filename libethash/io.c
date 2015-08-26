/*
  This file is part of ethash.

  ethash is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  ethash is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ethash.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file io.c
 * @author Lefteris Karapetsas <lefteris@ethdev.com>
 * @date 2015
 */
#include <string.h>
#include <stdio.h>
#include <errno.h>

#include "io.h"
#include "internal.h"

static char* LAST_DAG_FILENAME[_MAX_PATH] = { 0 };

char* ethash_io_lastdag_filename()
{
	return &LAST_DAG_FILENAME[0];
}

enum ethash_io_rc ethash_io_prepare(
	char const* dirname,
	ethash_h256_t const seedhash,
	FILE** output_file,
	uint64_t file_size,
	bool force_create
)
{
	char mutable_name[DAG_MUTABLE_NAME_MAX_SIZE];
	enum ethash_io_rc ret = ETHASH_IO_FAIL;

#ifdef WIN32
	char buf[_MAX_PATH] = { 0 };
	sprintf(buf, "%s", dirname);
	if (strlen(buf) && buf[strlen(buf)-1] == '\\')
		buf[strlen(buf)-1] = '\0';
	dirname = buf;
#endif

	// reset errno before io calls
	errno = 0;

	// assert directory exists
	if (!ethash_mkdir(dirname)) {
		ETHASH_CRITICAL("Could not create the ethash directory");
		goto end;
	}

	ethash_io_mutable_name(ETHASH_REVISION, &seedhash, mutable_name);
	char* tmpfile = ethash_io_create_filename(dirname, mutable_name, strlen(mutable_name));
	if (!tmpfile) {
		ETHASH_CRITICAL("Could not create the full DAG pathname");
		goto end;
	}

	strcpy(LAST_DAG_FILENAME, tmpfile);

	FILE *f;
	if (!force_create) {
		// try to open the file
		f = ethash_fopen(tmpfile, "rb+");
		if (f) {
			size_t found_size;
			if (!ethash_file_size(f, &found_size)) {
				fclose(f);
				ETHASH_CRITICAL("Could not query size of DAG file: \"%s\"", tmpfile);
				goto free_memo;
			}
			if (file_size != found_size - ETHASH_DAG_MAGIC_NUM_SIZE) {
				fclose(f);
				ret = ETHASH_IO_MEMO_SIZE_MISMATCH;
				goto free_memo;
			}
			// compare the magic number, no need to care about endianess since it's local
			uint64_t magic_num;
			if (fread(&magic_num, ETHASH_DAG_MAGIC_NUM_SIZE, 1, f) != 1) {
				// I/O error
				fclose(f);
				ETHASH_CRITICAL("Could not read from DAG file: \"%s\"", tmpfile);
				ret = ETHASH_IO_MEMO_SIZE_MISMATCH;
				goto free_memo;
			}
			if (magic_num != ETHASH_DAG_MAGIC_NUM) {
				fclose(f);
				ret = ETHASH_IO_MEMO_SIZE_MISMATCH;
				goto free_memo;
			}
			ret = ETHASH_IO_MEMO_MATCH;
			goto set_file;
		}
	}
	
	// file does not exist, will need to be created
	f = ethash_fopen(tmpfile, "wb+");
	if (!f) {
		ETHASH_CRITICAL("Could not create DAG file: \"%s\"", tmpfile);
		goto free_memo;
	}
	// make sure it's of the proper size
	if (fseek(f, (long int)(file_size + ETHASH_DAG_MAGIC_NUM_SIZE - 1), SEEK_SET) != 0) {
		fclose(f);
		ETHASH_CRITICAL("Could not seek to the end of DAG file: \"%s\". Insufficient space?", tmpfile);
		goto free_memo;
	}
	if (fputc('\n', f) == EOF) {
		fclose(f);
		ETHASH_CRITICAL("Could not write in the end of DAG file: \"%s\". Insufficient space?", tmpfile);
		goto free_memo;
	}
	if (fflush(f) != 0) {
		fclose(f);
		ETHASH_CRITICAL("Could not flush at end of DAG file: \"%s\". Insufficient space?", tmpfile);
		goto free_memo;
	}
	ret = ETHASH_IO_MEMO_MISMATCH;
	goto set_file;

	ret = ETHASH_IO_MEMO_MATCH;
set_file:
	*output_file = f;
free_memo:
	free(tmpfile);
end:
	return ret;
}

enum ethash_io_rc ethash_iomem_prepare(
	ethash_full_t eth,
	ethash_h256_t const seedhash,
	size_t file_size
)
{
	enum ethash_io_rc ret = ETHASH_IO_FAIL;
#if 0
	// direct mem alloc size is too big for x86 windows binaries
#define SZ_08M ((size_t) 8 * 1024 * 1024)
#define SZ_16M ((size_t)16 * 1024 * 1024)
#define SZ_32M ((size_t)32 * 1024 * 1024)
#define SZ_64M ((size_t)64 * 1024 * 1024)
#define SZ_PAGE SZ_64M
	size_t blocs = (file_size / SZ_PAGE) + 1;
	for (size_t n=1; n < blocs; n++) {
		// progressive alloc seems required...
		eth->data = (node*) calloc(n, SZ_PAGE);
		free(eth->data);
	}
	eth->data = (node*) calloc(blocs, SZ_PAGE);
	//if (eth->data) {
		ret = ETHASH_IO_MEMO_MISMATCH;
		eth->ismem = true;
		eth->seed = seedhash;
	}
#endif

	eth->ismem = true;
	eth->seed = seedhash;
	ethash_io_mutable_name(ETHASH_REVISION, &seedhash, eth->mutable_name);
	ret = ETHASH_IO_MEMO_MISMATCH;

	return ret;
}

enum ethash_io_rc ethash_iomem_openexisting(
	char const* dirname,
	ethash_full_t const eth,
	FILE** output_file,
	uint64_t file_size
)
{
	enum ethash_io_rc ret = ETHASH_IO_FAIL;

	char* tmpfile = ethash_io_create_filename(dirname, eth->mutable_name, strlen(eth->mutable_name));
	if (!tmpfile) {
		goto end;
	}

	strcpy(LAST_DAG_FILENAME, tmpfile);

	// try to open the file
	FILE *f = ethash_fopen(tmpfile, "rb");
	if (f) {
		size_t found_size;
		if (!ethash_file_size(f, &found_size)) {
			fclose(f);
			ETHASH_CRITICAL("Could not query size of DAG file: \"%s\"", tmpfile);
			goto free_memo;
		}
		if (file_size != found_size - ETHASH_DAG_MAGIC_NUM_SIZE) {
			fclose(f);
			ret = ETHASH_IO_MEMO_SIZE_MISMATCH;
			goto free_memo;
		}
		ret = ETHASH_IO_MEMO_MATCH;
	}

set_file:
	*output_file = f;
free_memo:
	free(tmpfile);
end:
	return ret;
}
