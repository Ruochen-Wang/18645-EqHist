#include <stdlib.h>
#include <stdint.h>
#include <string>
#include <math.h>
#include <iostream>
#include <fstream>
#include <immintrin.h>
#include <malloc.h>
#include <omp.h>
//#include "opencv2/imgcodecs.hpp"

//using namespace cv;
//using namespace std;

#define WIDTH                  	512 
#define HEIGHT                  WIDTH
#define IMAGE_SIZE              WIDTH*HEIGHT
#define INTENSITY_SPACE         256
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

//void eq_hist(unsigned char *src, unsigned char *dst);
void openmp_cal_2hist(unsigned char *src0, unsigned char *src1, uint32_t *global_hist, uint32_t *global_diff);
void openmp_cal_hist(unsigned char *src, uint32_t *hist);
float compare_hist(unsigned int *H1, unsigned int *H2);

int main(){
    // prepare input
    std::ifstream input_file0("comp1.b", std::ifstream::binary); //pic1 for comparison
    std::ifstream input_file1("comp2.b", std::ifstream::binary); //pic2 for comparison
//    std::ofstream output_file("512_equalized.b", std::ios::out | std::ios::binary);

//    unsigned char *src = new unsigned char[IMAGE_SIZE];
	unsigned char *src0 = (unsigned char *)memalign(256, IMAGE_SIZE*sizeof(uint8_t));
    input_file0.read((char *)src0, IMAGE_SIZE);
    unsigned char *src1 = (unsigned char *)memalign(256, IMAGE_SIZE*sizeof(uint8_t));
    input_file1.read((char *)src1, IMAGE_SIZE);
    
	unsigned char *dst = new unsigned char [IMAGE_SIZE];
	uint32_t *hist0 = (uint32_t *)malloc(INTENSITY_SPACE*sizeof(uint32_t));
	uint32_t *hist1 = (uint32_t *)malloc(INTENSITY_SPACE*sizeof(uint32_t));
    uint32_t *global_diff = (uint32_t *)malloc(INTENSITY_SPACE*sizeof(uint32_t)); //modify
	

	int sum;
	unsigned long long t0, t1;	

	for(int i = 1; i <= 1; i++){
		printf("%2d ", i);

		// t0 = rdtsc();
		// openmp_cal_2hist(src0, src1, hist0, global_diff);
		// //openmp_cal_abs(global_diff, sum);
        // //abs+sum
		// t1 = rdtsc();
		// printf("%ld \n", t1-t0);



		t0 = rdtsc();
		openmp_cal_hist(src0, hist0);
		openmp_cal_hist(src1, hist1);
		compare_hist(hist0, hist1);
		t1 = rdtsc();
		printf("%ld\n", t1-t0);

	}
/*
	int hist1tt = 0, hist2tt = 0, hist5tt = 0;
	for(int i = 0; i < INTENSITY_SPACE; i++){
		hist1tt += hist1[i];
		hist5tt += hist5[i];
		if(hist1[i] != hist5[i])
			printf("at %d actually: %d, expected: %d\n", i, hist1[i], hist5[i]);	
	}
	printf("total actually: %d, expected: %d\n", hist1tt, hist5tt);
*/
//	for(int i = 0; i < INTENSITY_SPACE; i++) printf("hist[%d]: %d\n", i, hist[i]);
//    output_file.write((char *)dst, IMAGE_SIZE);
	free(hist0);
	free(hist1);
	free(src0);
    free(src1);
    return 0;
}

float compare_hist(unsigned int *H1, unsigned int *H2){
    // TODO: 
    //1. normalize H1 and H2 by dividing the IMAGE_SIZE.
    //2. calculate distance of two array H1 and H2. (L2 distance)
    double result = 0;
    double h1, h2;

    for (int i = 0; i < INTENSITY_SPACE; ++i){
        h1 = H1[i] / IMAGE_SIZE;
        h2 = H2[i] / IMAGE_SIZE;
        result += pow((H1[i] - H2[i]), 2);
    }
    return sqrt(result);
}



/*global hist is the histogram for src0*/
void openmp_cal_2hist(unsigned char *src0, unsigned char *src1, uint32_t *global_hist, uint32_t *global_diff){
    // collect histogram
	#pragma omp parallel num_threads(16)
	{
    __m256i a, b0, c0, d, e0, f0, g0;
	uint64_t a0, a1, a2, a3;
	uint64_t d0, d1, d2, d3;	
	uint32_t *hist = (uint32_t *)malloc(INTENSITY_SPACE*sizeof(uint32_t));
	uint32_t *local_diff= (uint32_t *)malloc(INTENSITY_SPACE*sizeof(uint32_t));
	int num_threads = omp_get_num_threads();
	int id = omp_get_thread_num();
	int work_size = IMAGE_SIZE/32/num_threads;
	int start = id*work_size;
	//printf("thread %d: %d to %d\n", id, start, start+work_size-1);
	//#pragma omp parallel for 
	for(int i = start; i < start+work_size; i += 1){
		a = _mm256_stream_load_si256 ((const __m256i*)src0+i);
		d = _mm256_stream_load_si256 ((const __m256i*)src1+i);
		// extract 64 bits from 256-bit
		a0 = _mm256_extract_epi64(a, 0);
		a1 = _mm256_extract_epi64(a, 1);
		a2 = _mm256_extract_epi64(a, 2);
		a3 = _mm256_extract_epi64(a, 3);
		d0 = _mm256_extract_epi64(d, 0);
		d1 = _mm256_extract_epi64(d, 1);
		d2 = _mm256_extract_epi64(d, 2);
		d3 = _mm256_extract_epi64(d, 3);

		
		local_diff[a0&0xFF]++;
		local_diff[(a0>>8)&0xFF]++;
		local_diff[(a0>>16)&0xFF]++;
		local_diff[(a0>>24)&0xFF]++;
		local_diff[(a0>>32)&0xFF]++;
		local_diff[(a0>>40)&0xFF]++;
		local_diff[(a0>>48)&0xFF]++;
		local_diff[(a0>>56)&0xFF]++;
		local_diff[a1&0xFF]++;
		local_diff[(a1>>8)&0xFF]++;
		local_diff[(a1>>16)&0xFF]++;
		local_diff[(a1>>24)&0xFF]++;
		local_diff[(a1>>32)&0xFF]++;
		local_diff[(a1>>40)&0xFF]++;
		local_diff[(a1>>48)&0xFF]++;
		local_diff[(a1>>56)&0xFF]++;
		local_diff[a2&0xFF]++;
		local_diff[(a2>>8)&0xFF]++;
		local_diff[(a2>>16)&0xFF]++;
		local_diff[(a2>>24)&0xFF]++;
		local_diff[(a2>>32)&0xFF]++;
		local_diff[(a2>>40)&0xFF]++;
		local_diff[(a2>>48)&0xFF]++;
		local_diff[(a2>>56)&0xFF]++;
		local_diff[a3&0xFF]++;
		local_diff[(a3>>8)&0xFF]++;
		local_diff[(a3>>16)&0xFF]++;
		local_diff[(a3>>24)&0xFF]++;
		local_diff[(a3>>32)&0xFF]++;
		local_diff[(a3>>40)&0xFF]++;
		local_diff[(a3>>48)&0xFF]++;
		local_diff[(a3>>56)&0xFF]++;


		local_diff[d0&0xFF]--;
		local_diff[(d0>>8)&0xFF]--;
		local_diff[(d0>>16)&0xFF]--;
		local_diff[(d0>>24)&0xFF]--;
		local_diff[(d0>>32)&0xFF]--;
		local_diff[(d0>>40)&0xFF]--;
		local_diff[(d0>>48)&0xFF]--;
		local_diff[(d0>>56)&0xFF]--;
		local_diff[d1&0xFF]--;
		local_diff[(d1>>8)&0xFF]--;
		local_diff[(d1>>16)&0xFF]--;
		local_diff[(d1>>24)&0xFF]--;
		local_diff[(d1>>32)&0xFF]--;
		local_diff[(d1>>40)&0xFF]--;
		local_diff[(d1>>48)&0xFF]--;
		local_diff[(d1>>56)&0xFF]--;
		local_diff[d2&0xFF]--;
		local_diff[(d2>>8)&0xFF]--;
		local_diff[(d2>>16)&0xFF]--;
		local_diff[(d2>>24)&0xFF]--;
		local_diff[(d2>>32)&0xFF]--;
		local_diff[(d2>>40)&0xFF]--;
		local_diff[(d2>>48)&0xFF]--;
		local_diff[(d2>>56)&0xFF]--;
		local_diff[d3&0xFF]--;
		local_diff[(d3>>8)&0xFF]--;
		local_diff[(d3>>16)&0xFF]--;
		local_diff[(d3>>24)&0xFF]--;
		local_diff[(d3>>32)&0xFF]--;
		local_diff[(d3>>40)&0xFF]--;
		local_diff[(d3>>48)&0xFF]--;
		local_diff[(d3>>56)&0xFF]--;



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



		#pragma omp critical
		{
			for(int i = 0; i < INTENSITY_SPACE; i+=8){	
				b0 = _mm256_loadu_si256((const __m256i*)(global_hist+i));
				c0 = _mm256_loadu_si256((const __m256i*)(hist+i));
				b0 = _mm256_add_epi32(b0, c0);
				_mm256_storeu_si256((__m256i*)(global_hist+i), b0);
				global_diff[i] += local_diff[i];
			}

		}

	}
}


void openmp_cal_hist(unsigned char *src, uint32_t *global_hist){
    // collect histogram
	#pragma omp parallel num_threads(16)
	{
    __m256i a, b0, c0;
	uint64_t a0, a1, a2, a3;
	uint32_t *hist = (uint32_t *)malloc(INTENSITY_SPACE*sizeof(uint32_t));
	int num_threads = omp_get_num_threads();
	int id = omp_get_thread_num();
	int work_size = IMAGE_SIZE/32/num_threads;
	int start = id*work_size;
	//printf("thread %d: %d to %d\n", id, start, start+work_size-1);
	//#pragma omp parallel for 
	for(int i = start; i < start+work_size; i += 1){
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
		#pragma omp critical
		{
			for(int i = 0; i < INTENSITY_SPACE; i+=8){	
				b0 = _mm256_loadu_si256((const __m256i*)(global_hist+i));
				c0 = _mm256_loadu_si256((const __m256i*)(hist+i));
				b0 = _mm256_add_epi32(b0, c0);
				_mm256_storeu_si256((__m256i*)(global_hist+i), b0);
			}
		}
	}
}





/*
void openmp_cal_abs(uint32_t *global_diff, int sum){
	int sum;
	__m256i a;
	__m256i s;
	unsigned int num_thread = 16;
	unsigned int size = (INTENSITY_SPACE-2) / num_thread;
	omp_set_num_threads(num_thread);

	#pragma omp parallel
	{
		unsigned int id = omp_get_thread_num();
		unsigned int start = id * size;
		unsigned int end = (id+1) * size;

		unsigned int local_sum = 0;

		for(int i = 0; i < start; i<end; i++){	
			a = _mm256_abs_epi32(global_diff+i);
			s = _mm256_hadd_epi32(a, a);
			local_sum = ((int*)&s)[0] + ((int*)&s)[1] + ((int*)&s)[4] + ((int*)&s)[5] + local_sum;
		}
	#pragma omp critical
	sum += local_sum;

	}
}
*/