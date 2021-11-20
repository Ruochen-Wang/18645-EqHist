#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <string>
#include <mpi.h>
#include <math.h>
#include <immintrin.h>
#include <malloc.h>

#define WIDTH                  	512 
#define HEIGHT                  WIDTH
#define IMAGE_SIZE              WIDTH*HEIGHT
#define INTENSITY_SPACE         256
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

void mpi_cal_hist(unsigned char *src, uint32_t *hist, int id);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	MPI_File input;
//    std::ifstream input_file("512.b", std::ifstream::binary);
	MPI_File_open(MPI_COMM_WORLD, "512.b", MPI_MODE_RDONLY, MPI_INFO_NULL, &input);
	unsigned char *src = (unsigned char *)memalign(256, IMAGE_SIZE*sizeof(uint8_t));
//    input_file.read((char *)src, IMAGE_SIZE);
	MPI_File_read_all(input, src, IMAGE_SIZE, MPI_UNSIGNED, MPI_STATUS_IGNORE);
 
	uint32_t *local_hist = (uint32_t *)memalign(32, INTENSITY_SPACE*sizeof(uint32_t));
	uint32_t *global_hist = (uint32_t *)memalign(32, INTENSITY_SPACE*sizeof(uint32_t));
	printf("proc %d: %p, %p, %p\n", world_rank, local_hist, global_hist, &input);
	
	unsigned long long t0, t1;
	if(world_rank == 0){
		t0 = rdtsc();
		mpi_cal_hist(src, local_hist, world_rank);
		MPI_Reduce(local_hist, global_hist, INTENSITY_SPACE, MPI_UNSIGNED, MPI_SUM, 0, MPI_COMM_WORLD);
		t1 = rdtsc(); 
		printf("proc 0 time: %lld\n", t1-t0);
	}
	else{
		mpi_cal_hist(src, local_hist, world_rank);
		MPI_Reduce(local_hist, global_hist, INTENSITY_SPACE, MPI_UNSIGNED, MPI_SUM, 0, MPI_COMM_WORLD);
	}
	free(src);
	free(local_hist);	
	free(global_hist);

	MPI_File_close(&input);
    MPI_Finalize();   
	return 0;
}



void mpi_cal_hist(unsigned char *src, uint32_t *hist, int id){
    // collect histogram
    __m256i a;
	uint64_t a0, a1, a2, a3;
	for(int i = id*1024; i < (id+1)*1024; i += 1){
		a = _mm256_stream_load_si256 ((const __m256i*)src+i);
		// extract 64 bits from 256-bit
		a0 = _mm256_extract_epi64(a, 0);
		a1 = _mm256_extract_epi64(a, 1);
		a2 = _mm256_extract_epi64(a, 2);
		a3 = _mm256_extract_epi64(a, 3);
		// extracrt 8 bits from 64-bit
		hist[a0&0xFF]++;
		hist[(a0>>8)&0xFF]++;
		hist[(a0>>16)&0xFF]++;
		hist[(a0>>24)&0xFF]++;
		hist[(a0>>32)&0xFF]++;
		hist[(a0>>40)&0xFF]++;
		hist[(a0>>48)&0xFF]++;
		hist[(a0>>56)&0xFF]++;
		hist[a1&0xFF]++;
		hist[(a1>>8)&0xFF]++;
		hist[(a1>>16)&0xFF]++;
		hist[(a1>>24)&0xFF]++;
		hist[(a1>>32)&0xFF]++;
		hist[(a1>>40)&0xFF]++;
		hist[(a1>>48)&0xFF]++;
		hist[(a1>>56)&0xFF]++;
		hist[a2&0xFF]++;
		hist[(a2>>8)&0xFF]++;
		hist[(a2>>16)&0xFF]++;
		hist[(a2>>24)&0xFF]++;
		hist[(a2>>32)&0xFF]++;
		hist[(a2>>40)&0xFF]++;
		hist[(a2>>48)&0xFF]++;
		hist[(a2>>56)&0xFF]++;
		hist[a3&0xFF]++;
		hist[(a3>>8)&0xFF]++;
		hist[(a3>>16)&0xFF]++;
		hist[(a3>>24)&0xFF]++;
		hist[(a3>>32)&0xFF]++;
		hist[(a3>>40)&0xFF]++;
		hist[(a3>>48)&0xFF]++;
		hist[(a3>>56)&0xFF]++;
	}
}
