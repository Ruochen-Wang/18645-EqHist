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

void openmp_cal_2hist(unsigned char *src0, unsigned char *src1, uint32_t *global_hist0, uint32_t *global_hist2, int *global_diff);
float openmp_cal_abs_simd(int *global_diff);
int main(){
    // prepare input
    std::ifstream input_file("512.b", std::ifstream::binary);
//    std::ofstream output_file("512_equalized.b", std::ios::out | std::ios::binary);

//    unsigned char *src = new unsigned char[IMAGE_SIZE];
	unsigned char *src0 = (unsigned char *)memalign(256, IMAGE_SIZE*sizeof(uint8_t));
	unsigned char *src1 = (unsigned char *)memalign(256, IMAGE_SIZE*sizeof(uint8_t));
    input_file.read((char *)src, IMAGE_SIZE);
    
	unsigned char *dst = new unsigned char [IMAGE_SIZE];
	uint32_t *hist0 = (uint32_t *)malloc(INTENSITY_SPACE*sizeof(uint32_t));
	uint32_t *hist1 = (uint32_t *)malloc(INTENSITY_SPACE*sizeof(uint32_t));
	float *lut1 = (float *)malloc(INTENSITY_SPACE*sizeof(float));
    int *global_diff = (int *)malloc(INTENSITY_SPACE*sizeof(int)); //modify
	

	unsigned long long t0, t1;	

	for(int i = 1; i <= 30; i++){
		printf("%2d ", i);

		t0 = rdtsc();
		openmp_cal_2hist(src0, src1, hist0, hist1, global_diff);
		cal_lut_base(hist1, lut1);
		t1 = rdtsc();
		printf("%ld ", t1-t0);

	}
	free(hist1);
	free(hist2);
	free(lut1);
	free(src);
    return 0;
}

/*global hist is the histogram for src0*/
void openmp_cal_2hist(unsigned char *src0, unsigned char *src1, uint32_t *global_hist0, uint32_t *global_hist2, int *global_diff){
    // collect histogram
	#pragma omp parallel num_threads(16)
	{
    __m256i a, b0, c0, d, e0, f0, g0, h0;
	uint64_t a0, a1, a2, a3;
	uint64_t d0, d1, d2, d3;	
	uint32_t *hist1 = (uint32_t *)malloc(INTENSITY_SPACE*sizeof(uint32_t));
	uint32_t *hist2 = (uint32_t *)malloc(INTENSITY_SPACE*sizeof(uint32_t));
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


	
		// extracrt 8 bits from 64-bit
		hist1[a0&0xFF]++;
		hist1[(a0>>8)&0xFF]++;
		hist1[(a0>>16)&0xFF]++;
		hist1[(a0>>24)&0xFF]++;
		hist1[(a0>>32)&0xFF]++;
		hist1[(a0>>40)&0xFF]++;
		hist1[(a0>>48)&0xFF]++;
		hist1[(a0>>56)&0xFF]++;
		hist1[a1&0xFF]++;
		hist1[(a1>>8)&0xFF]++;
		hist1[(a1>>16)&0xFF]++;
		hist1[(a1>>24)&0xFF]++;
		hist1[(a1>>32)&0xFF]++;
		hist1[(a1>>40)&0xFF]++;
		hist1[(a1>>48)&0xFF]++;
		hist1[(a1>>56)&0xFF]++;
		hist1[a2&0xFF]++;
		hist1[(a2>>8)&0xFF]++;
		hist1[(a2>>16)&0xFF]++;
		hist1[(a2>>24)&0xFF]++;
		hist1[(a2>>32)&0xFF]++;
		hist1[(a2>>40)&0xFF]++;
		hist1[(a2>>48)&0xFF]++;
		hist1[(a2>>56)&0xFF]++;
		hist1[a3&0xFF]++;
		hist1[(a3>>8)&0xFF]++;
		hist1[(a3>>16)&0xFF]++;
		hist1[(a3>>24)&0xFF]++;
		hist1[(a3>>32)&0xFF]++;
		hist1[(a3>>40)&0xFF]++;
		hist1[(a3>>48)&0xFF]++;
		hist1[(a3>>56)&0xFF]++;

        hist2[d0&0xFF]++;
		hist2[(d0>>8)&0xFF]++;
		hist2[(d0>>16)&0xFF]++;
		hist2[(d0>>24)&0xFF]++;
		hist2[(d0>>32)&0xFF]++;
		hist2[(d0>>40)&0xFF]++;
		hist2[(d0>>48)&0xFF]++;
		hist2[(d0>>56)&0xFF]++;
		hist2[d1&0xFF]++;
		hist2[(d1>>8)&0xFF]++;
		hist2[(d1>>16)&0xFF]++;
		hist2[(d1>>24)&0xFF]++;
		hist2[(d1>>32)&0xFF]++;
		hist2[(d1>>40)&0xFF]++;
		hist2[(d1>>48)&0xFF]++;
		hist2[(d1>>56)&0xFF]++;
		hist2[d2&0xFF]++;
		hist2[(d2>>8)&0xFF]++;
		hist2[(d2>>16)&0xFF]++;
		hist2[(d2>>24)&0xFF]++;
		hist2[(d2>>32)&0xFF]++;
		hist2[(d2>>40)&0xFF]++;
		hist2[(d2>>48)&0xFF]++;
		hist2[(d2>>56)&0xFF]++;
		hist2[d3&0xFF]++;
		hist2[(d3>>8)&0xFF]++;
		hist2[(d3>>16)&0xFF]++;
		hist2[(d3>>24)&0xFF]++;
		hist2[(d3>>32)&0xFF]++;
		hist2[(d3>>40)&0xFF]++;
		hist2[(d3>>48)&0xFF]++;
		hist2[(d3>>56)&0xFF]++;


	    }

		#pragma omp critical
		{
			for(int i = 0; i < INTENSITY_SPACE; i+=8){	
				b0 = _mm256_loadu_si256((const __m256i*)(global_hist1+i));
				c0 = _mm256_loadu_si256((const __m256i*)(hist1+i));
				b0 = _mm256_add_epi32(b0, c0);
				_mm256_storeu_si256((__m256i*)(global_hist1+i), b0);

				e0 = _mm256_loadu_si256((const __m256i*)(global_hist2+i));
				f0 = _mm256_loadu_si256((const __m256i*)(hist2+i));
				e0 = _mm256_add_epi32(e0, f0);
				_mm256_storeu_si256((__m256i*)(global_hist2+i), e0);

				g0 = _mm256_sub_epi32(c0, f0);
                h0 = _mm256_loadu_si256((const __m256i*)(global_diff+i));
                g0 = _mm256_add_epi32(h0, g0);
				_mm256_storeu_si256((__m256i*)(global_diff+i), g0); //modify
                //printf("diff: %d \n", global_diff[i]);
			}
		}

	}
}

// SIMD abs
float openmp_cal_abs_simd(int *global_diff){
    float sum = 0;
	//unsigned int num_thread = 16;
	//unsigned int size = (INTENSITY_SPACE) / num_thread;
	{
		uint32_t s[8] = {};
		__m256i a, b;
    	__m256i sum0;
		sum0 = _mm256_set1_epi32(0);

		for(int i = 0; i<INTENSITY_SPACE; i+=8){	
            a = _mm256_loadu_si256((const __m256i*)(global_diff+i));
			b = _mm256_abs_epi32(a);
			sum0 = _mm256_add_epi32(b, sum0);
		}
        _mm256_store_si256((__m256i*)s, sum0);
        for(int i=0; i<8; i++){
            sum = sum + s[i];
        }
	}
	return sum/IMAGE_SIZE;
}
