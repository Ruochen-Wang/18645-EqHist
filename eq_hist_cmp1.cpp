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
void openmp_cal_2hist(unsigned char *src0, unsigned char *src1, uint32_t *global_hist1, uint32_t *global_hist2, int *global_diff);
void eq_hist(unsigned char *src, unsigned char *dst);
void cal_lut(unsigned char *src, uint8_t *lut);
float compare_hist(unsigned int *H1, unsigned int *H2);
float openmp_cal_abs(int *global_diff);

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
    int *global_diff = (int *)malloc(INTENSITY_SPACE*sizeof(int)); //modify

	float sum;
	unsigned long long t0, t1;	

	for(int i = 1; i <= 30; i++){
		printf("%2d ", i);

		t0 = rdtsc();
		openmp_cal_2hist(src0, src1, hist0, hist1, global_diff);
		sum = openmp_cal_abs(global_diff);
        //abs+sum
		t1 = rdtsc();
		printf("delay: %ld \n", t1-t0);
        //printf("sum: %f \n", sum);

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
void openmp_cal_2hist(unsigned char *src0, unsigned char *src1, uint32_t *global_hist1, uint32_t *global_hist2, int *global_diff){
    // collect histogram
	#pragma omp parallel num_threads(16)
	{
    __m256i a, b0, c0, d, e0, f0, g0, h0;
	uint64_t a0, a1, a2, a3;
	uint64_t d0, d1, d2, d3;	
	uint32_t *hist1 = (uint32_t *)malloc(INTENSITY_SPACE*sizeof(uint32_t));
	uint32_t *hist2= (uint32_t *)malloc(INTENSITY_SPACE*sizeof(uint32_t));
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


float openmp_cal_abs(int *global_diff){
    uint32_t s[8] = {};
    float sum = 0;
	__m256i a, b;
    __m256i sum0, size;
    uint32_t x=0;
    sum0 = _mm256_set1_epi32(x);
    //size = _mm256_set1_epi32(IMAGE_SIZE);
    // for(int i=0; i<INTENSITY_SPACE/2; i++){
    //     sum = sum + global_diff[i];
    //     printf("diff: % \n", sum);
    // }

	// unsigned int num_thread = 16;
	// unsigned int size = (INTENSITY_SPACE-2) / num_thread;

	//#pragma omp parallel num_threads(num_thread)
	//{
		// unsigned int id = omp_get_thread_num();
		// unsigned int start = id * size;
		// unsigned int end = (id+1) * size;

		// unsigned int local_sum = 0;

        
        // for(int i=0; i<INTENSITY_SPACE; i++){
        //     global_diff[i] = global_diff[i]/IMAGE_SIZE;
        // }

        
		for(int i = 0; i<INTENSITY_SPACE; i+=8){	
            a = _mm256_loadu_si256((const __m256i*)(global_diff+i));
            //b = _mm256_max_epi32(_mm256_sub_epi32(_mm256_set1_epi32(0.0), a), a);
			b = _mm256_abs_epi32(a);
            _mm256_storeu_si256((__m256i*)(global_diff+i), b);
		}
        // _mm256_store_si256((__m256i*)s, sum0);
        // for(int i=0; i<8; i++){
        //     sum = sum + s[i];
        //     printf("s: %ld \n", s[i]);
        // }

        for(int i=0; i<INTENSITY_SPACE; i++)
            sum = sum + global_diff[i];
        
        //sum = sum / IMAGE_SIZE;
        
        //printf("result: %.6f \n", sum/IMAGE_SIZE);
        return sum/IMAGE_SIZE;
	// #pragma omp critical
	// sum += local_sum;

	//}
}


