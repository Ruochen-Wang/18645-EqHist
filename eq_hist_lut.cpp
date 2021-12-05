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
void multistream_cal_hist(unsigned char *src, uint32_t *hist);
void new_multistream_cal_hist(unsigned char *src, uint32_t *hist);
void singlestream_cal_hist(unsigned char *src, uint32_t *hist);
void openmp_cal_hist(unsigned char *src, uint32_t *hist);
void base_cal_hist(unsigned char *src, uint32_t *hist);

void eq_hist(unsigned char *src, unsigned char *dst, uint32_t *hist);
void cal_lut_base(uint32_t *hist, float *lut);
void cal_lut2(uint32_t *hist, float *lut);
void cal_lut4(uint32_t *hist, float *lut);
void cal_lut8(uint32_t *hist, float *lut);
uint8_t sat_cast(uint16_t scaled_number);

int main(){
    // prepare input
    std::ifstream input_file("512.b", std::ifstream::binary);
//    std::ofstream output_file("512_equalized.b", std::ios::out | std::ios::binary);

//    unsigned char *src = new unsigned char[IMAGE_SIZE];
	unsigned char *src = (unsigned char *)memalign(256, IMAGE_SIZE*sizeof(uint8_t));
    input_file.read((char *)src, IMAGE_SIZE);
    
	unsigned char *dst = new unsigned char [IMAGE_SIZE];
	uint32_t *hist1 = (uint32_t *)malloc(INTENSITY_SPACE*sizeof(uint32_t));
	uint32_t *hist2 = (uint32_t *)malloc(INTENSITY_SPACE*sizeof(uint32_t));
	uint32_t *hist3 = (uint32_t *)malloc(INTENSITY_SPACE*sizeof(uint32_t));
	uint32_t *hist4 = (uint32_t *)malloc(INTENSITY_SPACE*sizeof(uint32_t));
	float *lut1 = (float *)malloc(INTENSITY_SPACE*sizeof(float));
	float *lut2 = (float *)malloc(INTENSITY_SPACE*sizeof(float));
	float *lut3 = (float *)malloc(INTENSITY_SPACE*sizeof(float));
	float *lut4 = (float *)malloc(INTENSITY_SPACE*sizeof(float));

	unsigned long long t0, t1;	

	for(int i = 1; i <= 30; i++){
		printf("%2d ", i);

		t0 = rdtsc();
		openmp_cal_hist(src, hist1);
		cal_lut_base(hist1, lut1);
		t1 = rdtsc();
		printf("%ld ", t1-t0);

		t0 = rdtsc();
		openmp_cal_hist(src, hist2);
		cal_lut2(hist2, lut2);
		t1 = rdtsc();
		printf("%ld ", t1-t0);

		t0 = rdtsc();
		openmp_cal_hist(src, hist3);
		cal_lut4(hist3, lut3);
		t1 = rdtsc();
		printf("%ld ", t1-t0);

		t0 = rdtsc();
		openmp_cal_hist(src, hist4);
		cal_lut8(hist4, lut4);
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
	free(hist1);
	free(hist2);
	free(hist3);
	free(hist4);
	free(lut1);
	free(lut2);
	free(lut3);
	free(lut4);
	free(src);
    return 0;
}


/*
void eq_hist(unsigned char *src, unsigned char *dst){
    uint8_t *lut = new uint8_t[INTENSITY_SPACE];
    cal_lut(src, lut);
    
    for (int i = 0; i < IMAGE_SIZE; ++i) {
        dst[i] = lut[(unsigned char)src[i]];
    }

}
*/
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

/*
*   @input: src: pointer to an array of IMAGE_SIZE number of unsigned char
*   @output: lut: pointer to an array of INTENSITY_SPACE number of uint8_t
*/
void new_multistream_cal_hist(unsigned char *src, uint32_t *hist){
    // collect histogram
    __m256i a, b, c, d, e, f, g, h;
	uint64_t a0, a1, a2, a3;
	uint64_t b0, b1, b2, b3;
	uint64_t c0, c1, c2, c3;
	uint64_t d0, d1, d2, d3;
	uint64_t e0, e1, e2, e3;
	uint64_t f0, f1, f2, f3;
	uint64_t g0, g1, g2, g3;
	uint64_t h0, h1, h2, h3;
	// divide into 4 streams, load 32 bytes at once
	for(int i = 0; i < IMAGE_SIZE/(32*4); i += 1){

		a = _mm256_stream_load_si256 ((__m256i const*)src+i);
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

		b = _mm256_stream_load_si256 ((__m256i const*)src+i+1*1024*2);
		b0 = _mm256_extract_epi64(b, 0);
		b1 = _mm256_extract_epi64(b, 1);
		b2 = _mm256_extract_epi64(b, 2);
		b3 = _mm256_extract_epi64(b, 3);
		hist[b0&0xFF]++;
		hist[(b0>>8)&0xFF]++;
		hist[(b0>>16)&0xFF]++;
		hist[(b0>>24)&0xFF]++;
		hist[(b0>>32)&0xFF]++;
		hist[(b0>>40)&0xFF]++;
		hist[(b0>>48)&0xFF]++;
		hist[(b0>>56)&0xFF]++;
		hist[b1&0xFF]++;
		hist[(b1>>8)&0xFF]++;
		hist[(b1>>16)&0xFF]++;
		hist[(b1>>24)&0xFF]++;
		hist[(b1>>32)&0xFF]++;
		hist[(b1>>40)&0xFF]++;
		hist[(b1>>48)&0xFF]++;
		hist[(b1>>56)&0xFF]++;
		hist[b2&0xFF]++;
		hist[(b2>>8)&0xFF]++;
		hist[(b2>>16)&0xFF]++;
		hist[(b2>>24)&0xFF]++;
		hist[(b2>>32)&0xFF]++;
		hist[(b2>>40)&0xFF]++;
		hist[(b2>>48)&0xFF]++;
		hist[(b2>>56)&0xFF]++;
		hist[b3&0xFF]++;
		hist[(b3>>8)&0xFF]++;
		hist[(b3>>16)&0xFF]++;
		hist[(b3>>24)&0xFF]++;
		hist[(b3>>32)&0xFF]++;
		hist[(b3>>40)&0xFF]++;
		hist[(b3>>48)&0xFF]++;
		hist[(b3>>56)&0xFF]++;

		c = _mm256_stream_load_si256 ((__m256i const*)src+i+2*1024*2);
		c0 = _mm256_extract_epi64(c, 0);
		c1 = _mm256_extract_epi64(c, 1);
		c2 = _mm256_extract_epi64(c, 2);
		c3 = _mm256_extract_epi64(c, 3);
		hist[c0&0xFF]++;
		hist[(c0>>8)&0xFF]++;
		hist[(c0>>16)&0xFF]++;
		hist[(c0>>24)&0xFF]++;
		hist[(c0>>32)&0xFF]++;
		hist[(c0>>40)&0xFF]++;
		hist[(c0>>48)&0xFF]++;
		hist[(c0>>56)&0xFF]++;
		hist[c1&0xFF]++;
		hist[(c1>>8)&0xFF]++;
		hist[(c1>>16)&0xFF]++;
		hist[(c1>>24)&0xFF]++;
		hist[(c1>>32)&0xFF]++;
		hist[(c1>>40)&0xFF]++;
		hist[(c1>>48)&0xFF]++;
		hist[(c1>>56)&0xFF]++;
		hist[c2&0xFF]++;
		hist[(c2>>8)&0xFF]++;
		hist[(c2>>16)&0xFF]++;
		hist[(c2>>24)&0xFF]++;
		hist[(c2>>32)&0xFF]++;
		hist[(c2>>40)&0xFF]++;
		hist[(c2>>48)&0xFF]++;
		hist[(c2>>56)&0xFF]++;
		hist[c3&0xFF]++;
		hist[(c3>>8)&0xFF]++;
		hist[(c3>>16)&0xFF]++;
		hist[(c3>>24)&0xFF]++;
		hist[(c3>>32)&0xFF]++;
		hist[(c3>>40)&0xFF]++;
		hist[(c3>>48)&0xFF]++;
		hist[(c3>>56)&0xFF]++;

		d = _mm256_stream_load_si256 ((__m256i const*)src+i+3*1024*2);
		d0 = _mm256_extract_epi64(d, 0);
		d1 = _mm256_extract_epi64(d, 1);
		d2 = _mm256_extract_epi64(d, 2);
		d3 = _mm256_extract_epi64(d, 3);
		hist[d0&0xFF]++;
		hist[(d0>>8)&0xFF]++;
		hist[(d0>>16)&0xFF]++;
		hist[(d0>>24)&0xFF]++;
		hist[(d0>>32)&0xFF]++;
		hist[(d0>>40)&0xFF]++;
		hist[(d0>>48)&0xFF]++;
		hist[(d0>>56)&0xFF]++;
		hist[d1&0xFF]++;
		hist[(d1>>8)&0xFF]++;
		hist[(d1>>16)&0xFF]++;
		hist[(d1>>24)&0xFF]++;
		hist[(d1>>32)&0xFF]++;
		hist[(d1>>40)&0xFF]++;
		hist[(d1>>48)&0xFF]++;
		hist[(d1>>56)&0xFF]++;
		hist[d2&0xFF]++;
		hist[(d2>>8)&0xFF]++;
		hist[(d2>>16)&0xFF]++;
		hist[(d2>>24)&0xFF]++;
		hist[(d2>>32)&0xFF]++;
		hist[(d2>>40)&0xFF]++;
		hist[(d2>>48)&0xFF]++;
		hist[(d2>>56)&0xFF]++;
		hist[d3&0xFF]++;
		hist[(d3>>8)&0xFF]++;
		hist[(d3>>16)&0xFF]++;
		hist[(d3>>24)&0xFF]++;
		hist[(d3>>32)&0xFF]++;
		hist[(d3>>40)&0xFF]++;
		hist[(d3>>48)&0xFF]++;
		hist[(d3>>56)&0xFF]++;
	}
}

void multistream_cal_hist(unsigned char *src, uint32_t *hist){
    // collect histogram
    __m256i a, b, c, d, e, f, g, h;
	uint64_t a0, a1, a2, a3;
	uint64_t b0, b1, b2, b3;
	uint64_t c0, c1, c2, c3;
	uint64_t d0, d1, d2, d3;
	uint64_t e0, e1, e2, e3;
	uint64_t f0, f1, f2, f3;
	uint64_t g0, g1, g2, g3;
	uint64_t h0, h1, h2, h3;
	// divide into 8 streams, load 32 bytes at once
	// TODO: fix loop size
	for(int i = 0; i < IMAGE_SIZE/(32*8); i += 1){
		a = _mm256_stream_load_si256 ((__m256i const*)src+i);
		b = _mm256_stream_load_si256 ((__m256i const*)src+i+1*1024);
		c = _mm256_stream_load_si256 ((__m256i const*)src+i+2*1024);
		d = _mm256_stream_load_si256 ((__m256i const*)src+i+3*1024);
		e = _mm256_stream_load_si256 ((__m256i const*)src+i+4*1024);
		f = _mm256_stream_load_si256 ((__m256i const*)src+i+5*1024);
		g = _mm256_stream_load_si256 ((__m256i const*)src+i+6*1024);
		h = _mm256_stream_load_si256 ((__m256i const*)src+i+7*1024);

		// extract 64 bits from 256-bit
		a0 = _mm256_extract_epi64(a, 0);
		a1 = _mm256_extract_epi64(a, 1);
		a2 = _mm256_extract_epi64(a, 2);
		a3 = _mm256_extract_epi64(a, 3);
		b0 = _mm256_extract_epi64(b, 0);
		b1 = _mm256_extract_epi64(b, 1);
		b2 = _mm256_extract_epi64(b, 2);
		b3 = _mm256_extract_epi64(b, 3);
		c0 = _mm256_extract_epi64(c, 0);
		c1 = _mm256_extract_epi64(c, 1);
		c2 = _mm256_extract_epi64(c, 2);
		c3 = _mm256_extract_epi64(c, 3);
		d0 = _mm256_extract_epi64(d, 0);
		d1 = _mm256_extract_epi64(d, 1);
		d2 = _mm256_extract_epi64(d, 2);
		d3 = _mm256_extract_epi64(d, 3);

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

		hist[b0&0xFF]++;
		hist[(b0>>8)&0xFF]++;
		hist[(b0>>16)&0xFF]++;
		hist[(b0>>24)&0xFF]++;
		hist[(b0>>32)&0xFF]++;
		hist[(b0>>40)&0xFF]++;
		hist[(b0>>48)&0xFF]++;
		hist[(b0>>56)&0xFF]++;
		hist[b1&0xFF]++;
		hist[(b1>>8)&0xFF]++;
		hist[(b1>>16)&0xFF]++;
		hist[(b1>>24)&0xFF]++;
		hist[(b1>>32)&0xFF]++;
		hist[(b1>>40)&0xFF]++;
		hist[(b1>>48)&0xFF]++;
		hist[(b1>>56)&0xFF]++;
		hist[b2&0xFF]++;
		hist[(b2>>8)&0xFF]++;
		hist[(b2>>16)&0xFF]++;
		hist[(b2>>24)&0xFF]++;
		hist[(b2>>32)&0xFF]++;
		hist[(b2>>40)&0xFF]++;
		hist[(b2>>48)&0xFF]++;
		hist[(b2>>56)&0xFF]++;
		hist[b3&0xFF]++;
		hist[(b3>>8)&0xFF]++;
		hist[(b3>>16)&0xFF]++;
		hist[(b3>>24)&0xFF]++;
		hist[(b3>>32)&0xFF]++;
		hist[(b3>>40)&0xFF]++;
		hist[(b3>>48)&0xFF]++;
		hist[(b3>>56)&0xFF]++;

		hist[c0&0xFF]++;
		hist[(c0>>8)&0xFF]++;
		hist[(c0>>16)&0xFF]++;
		hist[(c0>>24)&0xFF]++;
		hist[(c0>>32)&0xFF]++;
		hist[(c0>>40)&0xFF]++;
		hist[(c0>>48)&0xFF]++;
		hist[(c0>>56)&0xFF]++;
		hist[c1&0xFF]++;
		hist[(c1>>8)&0xFF]++;
		hist[(c1>>16)&0xFF]++;
		hist[(c1>>24)&0xFF]++;
		hist[(c1>>32)&0xFF]++;
		hist[(c1>>40)&0xFF]++;
		hist[(c1>>48)&0xFF]++;
		hist[(c1>>56)&0xFF]++;
		hist[c2&0xFF]++;
		hist[(c2>>8)&0xFF]++;
		hist[(c2>>16)&0xFF]++;
		hist[(c2>>24)&0xFF]++;
		hist[(c2>>32)&0xFF]++;
		hist[(c2>>40)&0xFF]++;
		hist[(c2>>48)&0xFF]++;
		hist[(c2>>56)&0xFF]++;
		hist[c3&0xFF]++;
		hist[(c3>>8)&0xFF]++;
		hist[(c3>>16)&0xFF]++;
		hist[(c3>>24)&0xFF]++;
		hist[(c3>>32)&0xFF]++;
		hist[(c3>>40)&0xFF]++;
		hist[(c3>>48)&0xFF]++;
		hist[(c3>>56)&0xFF]++;

		hist[d0&0xFF]++;
		hist[(d0>>8)&0xFF]++;
		hist[(d0>>16)&0xFF]++;
		hist[(d0>>24)&0xFF]++;
		hist[(d0>>32)&0xFF]++;
		hist[(d0>>40)&0xFF]++;
		hist[(d0>>48)&0xFF]++;
		hist[(d0>>56)&0xFF]++;
		hist[d1&0xFF]++;
		hist[(d1>>8)&0xFF]++;
		hist[(d1>>16)&0xFF]++;
		hist[(d1>>24)&0xFF]++;
		hist[(d1>>32)&0xFF]++;
		hist[(d1>>40)&0xFF]++;
		hist[(d1>>48)&0xFF]++;
		hist[(d1>>56)&0xFF]++;
		hist[d2&0xFF]++;
		hist[(d2>>8)&0xFF]++;
		hist[(d2>>16)&0xFF]++;
		hist[(d2>>24)&0xFF]++;
		hist[(d2>>32)&0xFF]++;
		hist[(d2>>40)&0xFF]++;
		hist[(d2>>48)&0xFF]++;
		hist[(d2>>56)&0xFF]++;
		hist[d3&0xFF]++;
		hist[(d3>>8)&0xFF]++;
		hist[(d3>>16)&0xFF]++;
		hist[(d3>>24)&0xFF]++;
		hist[(d3>>32)&0xFF]++;
		hist[(d3>>40)&0xFF]++;
		hist[(d3>>48)&0xFF]++;
		hist[(d3>>56)&0xFF]++;

	
		e0 = _mm256_extract_epi64(e, 0);
		e1 = _mm256_extract_epi64(e, 1);
		e2 = _mm256_extract_epi64(e, 2);
		e3 = _mm256_extract_epi64(e, 3);
		f0 = _mm256_extract_epi64(f, 0);
		f1 = _mm256_extract_epi64(f, 1);
		f2 = _mm256_extract_epi64(f, 2);
		f3 = _mm256_extract_epi64(f, 3);
		g0 = _mm256_extract_epi64(g, 0);
		g1 = _mm256_extract_epi64(g, 1);
		g2 = _mm256_extract_epi64(g, 2);
		g3 = _mm256_extract_epi64(g, 3);
		h0 = _mm256_extract_epi64(h, 0);
		h1 = _mm256_extract_epi64(h, 1);
		h2 = _mm256_extract_epi64(h, 2);
		h3 = _mm256_extract_epi64(h, 3);
	
		hist[e0&0xFF]++;
		hist[(e0>>8)&0xFF]++;
		hist[(e0>>16)&0xFF]++;
		hist[(e0>>24)&0xFF]++;
		hist[(e0>>32)&0xFF]++;
		hist[(e0>>40)&0xFF]++;
		hist[(e0>>48)&0xFF]++;
		hist[(e0>>56)&0xFF]++;
		hist[e1&0xFF]++;
		hist[(e1>>8)&0xFF]++;
		hist[(e1>>16)&0xFF]++;
		hist[(e1>>24)&0xFF]++;
		hist[(e1>>32)&0xFF]++;
		hist[(e1>>40)&0xFF]++;
		hist[(e1>>48)&0xFF]++;
		hist[(e1>>56)&0xFF]++;
		hist[e2&0xFF]++;
		hist[(e2>>8)&0xFF]++;
		hist[(e2>>16)&0xFF]++;
		hist[(e2>>24)&0xFF]++;
		hist[(e2>>32)&0xFF]++;
		hist[(e2>>40)&0xFF]++;
		hist[(e2>>48)&0xFF]++;
		hist[(e2>>56)&0xFF]++;
		hist[e3&0xFF]++;
		hist[(e3>>8)&0xFF]++;
		hist[(e3>>16)&0xFF]++;
		hist[(e3>>24)&0xFF]++;
		hist[(e3>>32)&0xFF]++;
		hist[(e3>>40)&0xFF]++;
		hist[(e3>>48)&0xFF]++;
		hist[(e3>>56)&0xFF]++;

		hist[f0&0xFF]++;
		hist[(f0>>8)&0xFF]++;
		hist[(f0>>16)&0xFF]++;
		hist[(f0>>24)&0xFF]++;
		hist[(f0>>32)&0xFF]++;
		hist[(f0>>40)&0xFF]++;
		hist[(f0>>48)&0xFF]++;
		hist[(f0>>56)&0xFF]++;
		hist[f1&0xFF]++;
		hist[(f1>>8)&0xFF]++;
		hist[(f1>>16)&0xFF]++;
		hist[(f1>>24)&0xFF]++;
		hist[(f1>>32)&0xFF]++;
		hist[(f1>>40)&0xFF]++;
		hist[(f1>>48)&0xFF]++;
		hist[(f1>>56)&0xFF]++;
		hist[f2&0xFF]++;
		hist[(f2>>8)&0xFF]++;
		hist[(f2>>16)&0xFF]++;
		hist[(f2>>24)&0xFF]++;
		hist[(f2>>32)&0xFF]++;
		hist[(f2>>40)&0xFF]++;
		hist[(f2>>48)&0xFF]++;
		hist[(f2>>56)&0xFF]++;
		hist[f3&0xFF]++;
		hist[(f3>>8)&0xFF]++;
		hist[(f3>>16)&0xFF]++;
		hist[(f3>>24)&0xFF]++;
		hist[(f3>>32)&0xFF]++;
		hist[(f3>>40)&0xFF]++;
		hist[(f3>>48)&0xFF]++;
		hist[(f3>>56)&0xFF]++;

		hist[g0&0xFF]++;
		hist[(g0>>8)&0xFF]++;
		hist[(g0>>16)&0xFF]++;
		hist[(g0>>24)&0xFF]++;
		hist[(g0>>32)&0xFF]++;
		hist[(g0>>40)&0xFF]++;
		hist[(g0>>48)&0xFF]++;
		hist[(g0>>56)&0xFF]++;
		hist[g1&0xFF]++;
		hist[(g1>>8)&0xFF]++;
		hist[(g1>>16)&0xFF]++;
		hist[(g1>>24)&0xFF]++;
		hist[(g1>>32)&0xFF]++;
		hist[(g1>>40)&0xFF]++;
		hist[(g1>>48)&0xFF]++;
		hist[(g1>>56)&0xFF]++;
		hist[g2&0xFF]++;
		hist[(g2>>8)&0xFF]++;
		hist[(g2>>16)&0xFF]++;
		hist[(g2>>24)&0xFF]++;
		hist[(g2>>32)&0xFF]++;
		hist[(g2>>40)&0xFF]++;
		hist[(g2>>48)&0xFF]++;
		hist[(g2>>56)&0xFF]++;
		hist[g3&0xFF]++;
		hist[(g3>>8)&0xFF]++;
		hist[(g3>>16)&0xFF]++;
		hist[(g3>>24)&0xFF]++;
		hist[(g3>>32)&0xFF]++;
		hist[(g3>>40)&0xFF]++;
		hist[(g3>>48)&0xFF]++;
		hist[(g3>>56)&0xFF]++;

		hist[h0&0xFF]++;
		hist[(h0>>8)&0xFF]++;
		hist[(h0>>16)&0xFF]++;
		hist[(h0>>24)&0xFF]++;
		hist[(h0>>32)&0xFF]++;
		hist[(h0>>40)&0xFF]++;
		hist[(h0>>48)&0xFF]++;
		hist[(h0>>56)&0xFF]++;
		hist[h1&0xFF]++;
		hist[(h1>>8)&0xFF]++;
		hist[(h1>>16)&0xFF]++;
		hist[(h1>>24)&0xFF]++;
		hist[(h1>>32)&0xFF]++;
		hist[(h1>>40)&0xFF]++;
		hist[(h1>>48)&0xFF]++;
		hist[(h1>>56)&0xFF]++;
		hist[h2&0xFF]++;
		hist[(h2>>8)&0xFF]++;
		hist[(h2>>16)&0xFF]++;
		hist[(h2>>24)&0xFF]++;
		hist[(h2>>32)&0xFF]++;
		hist[(h2>>40)&0xFF]++;
		hist[(h2>>48)&0xFF]++;
		hist[(h2>>56)&0xFF]++;
		hist[h3&0xFF]++;
		hist[(h3>>8)&0xFF]++;
		hist[(h3>>16)&0xFF]++;
		hist[(h3>>24)&0xFF]++;
		hist[(h3>>32)&0xFF]++;
		hist[(h3>>40)&0xFF]++;
		hist[(h3>>48)&0xFF]++;
		hist[(h3>>56)&0xFF]++;

	}
}
/*Verified correctness against baseline*/
void singlestream_cal_hist(unsigned char *src, uint32_t *hist){
    // collect histogram
    __m256i a;
	uint64_t a0, a1, a2, a3;
	// divide into 8 streams, load 32 bytes at once
	for(int i = 0; i < IMAGE_SIZE/32; i += 1){
		a = _mm256_stream_load_si256 ((const __m256i*)src+i);
//		a = _mm256_stream_load_si256 (reinterpret_cast<const __m256i*> (src+i));
//		printf("%d\n", i);
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
/*
		if(i==0){
			printf("%d, %d, %d, %d\n", a0&0xFF, (a0>>8)&0xFF, (a0>>16)&0xFF, (a0>>24)&0xFF);
			printf("%d, %d, %d, %d\n", a1&0xFF, (a1>>8)&0xFF, (a1>>16)&0xFF, (a1>>24)&0xFF);
			printf("%d, %d, %d, %d\n", a2&0xFF, (a2>>8)&0xFF, (a2>>16)&0xFF, (a2>>24)&0xFF);
			printf("%d, %d, %d, %d\n", a3&0xFF, (a3>>8)&0xFF, (a3>>16)&0xFF, (a3>>24)&0xFF);
		}
*/
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

/*global hist is the histogram for src0*/
void openmp_cal_2hist(unsigned char *src0, unsigned char *src1, uint32_t *global_hist){
    // collect histogram
	#pragma omp parallel num_threads(16)
	{
    __m256i a, b0, c0, d;
	uint64_t a0, a1, a2, a3;
	uint64_t d0, d1, d2, d3;	
	uint32_t *hist = (uint32_t *)malloc(INTENSITY_SPACE*sizeof(uint32_t));
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

void base_cal_hist(unsigned char *src, uint32_t *hist){
    for (int i = 0; i < IMAGE_SIZE; i++)
        hist[(unsigned char)src[i]]++;
}


void eq_hist(unsigned char *src, unsigned char *dst, uint32_t *hist){
    float *lut = new float[INTENSITY_SPACE];
    cal_lut8(hist, lut);
    
    for (int i = 0; i < IMAGE_SIZE; ++i) {
        dst[i] = sat_cast((uint16_t) lut[(unsigned char)src[i]]);
    }
}

void cal_lut_base(uint32_t *hist, float *lut){
    // find the first non-zero intensity
    int i = 0;
    while (!hist[i]) ++i;

    float scale = (INTENSITY_SPACE - 1.f)/(IMAGE_SIZE - hist[i]);
    int sum = 0;

    // find CDF
    for (lut[i++] = 0; i < INTENSITY_SPACE; ++i)
    {
        sum += hist[i];
        lut[i] = sum*scale; // prevent ovf
    }
}

/*
*   @input: src: pointer to an array of IMAGE_SIZE number of unsigned char
*   @output: lut: pointer to an array of INTENSITY_SPACE number of uint32_t
*/
void cal_lut2(uint32_t *hist, float *lut){
    // find the first non-zero intensity
    int fnz = 0;
    while (!hist[fnz]) ++fnz;

    float scale = (INTENSITY_SPACE - 1.f)/(IMAGE_SIZE - hist[fnz]);
    __m256 m_scale = _mm256_set1_ps(scale);

    // define SIMD variables
    __m256 l1, l2;
    __m256 pre_cdf1;

    // calculate 4 parts of sub-CDF
    int size = INTENSITY_SPACE / 2;
    // #pragma omp parallel for num_threads(8)
	for (int id = 0; id < 2; ++id)
	{
        // int id = omp_get_thread_num();
        int start = id * size;
        int end = (id + 1) * size;

        lut[start] = hist[start];
        for (int k = start+1; k < end; ++k) {
            lut[k] = hist[k] + lut[k - 1];
        }
    }

    pre_cdf1 = _mm256_set1_ps(lut[1 * size - 1]);
    for (int k = 0; k < size; k += 8) {
        l1 = _mm256_loadu_ps((float const*) &lut[0 * size + k]);
        l2 = _mm256_loadu_ps((float const*) &lut[1 * size + k]);

        l1 = _mm256_mul_ps(l1, m_scale);
        l2 = _mm256_add_ps(l2, pre_cdf1);

        _mm256_storeu_ps((float*) &lut[0 * size + k], l1);
        _mm256_storeu_ps((float*) &lut[1 * size + k], l2);
    }

    for (int k = 0; k < size; k += 8) {
        l1 = _mm256_loadu_ps((float const*) &lut[1 * size + k]);

        l1 = _mm256_mul_ps(l1, m_scale);

        _mm256_storeu_ps((float*) &lut[1 * size + k], l1);
    }
}


void cal_lut4(uint32_t *hist, float *lut){
    // find the first non-zero intensity
    int fnz = 0;
    while (!hist[fnz]) ++fnz;

    float scale = (INTENSITY_SPACE - 1.f)/(IMAGE_SIZE - hist[fnz]);
    __m256 m_scale = _mm256_set1_ps(scale);

    // define SIMD variables
    __m256 l1, l2, l3, l4;
    __m256 pre_cdf1, pre_cdf2;

    // calculate 4 parts of sub-CDF
    int size = INTENSITY_SPACE / 4;
    // #pragma omp parallel num_threads(8)
	for (int id = 0; id < 4; ++id)
	{
        // int id = omp_get_thread_num();
        int start = id * size;
        int end = (id + 1) * size;

        lut[start] = hist[start];
        for (int k = start+1; k < end; ++k) {
            lut[k] = hist[k] + lut[k - 1];
        }
    }

    pre_cdf1 = _mm256_set1_ps(lut[1 * size - 1]);
    pre_cdf2 = _mm256_set1_ps(lut[3 * size - 1]);
    for (int k = 0; k < size; k += 8) {
        l1 = _mm256_loadu_ps((float const*) &lut[0 * size + k]);
        l2 = _mm256_loadu_ps((float const*) &lut[1 * size + k]);
        l3 = _mm256_loadu_ps((float const*) &lut[3 * size + k]);

        l1 = _mm256_mul_ps(l1, m_scale);
        l2 = _mm256_add_ps(l2, pre_cdf1);
        l3 = _mm256_add_ps(l3, pre_cdf2);

        _mm256_storeu_ps((float*) &lut[0 * size + k], l1);
        _mm256_storeu_ps((float*) &lut[1 * size + k], l2);
        _mm256_storeu_ps((float*) &lut[3 * size + k], l3);
    }

    pre_cdf1 = _mm256_set1_ps(lut[2 * size - 1]);
    for (int k = 0; k < size; k += 8) {
        l1 = _mm256_loadu_ps((float const*) &lut[1 * size + k]);
        l2 = _mm256_loadu_ps((float const*) &lut[2 * size + k]);
        l3 = _mm256_loadu_ps((float const*) &lut[3 * size + k]);

        l1 = _mm256_mul_ps(l1, m_scale);
        l2 = _mm256_add_ps(l2, pre_cdf1);
        l3 = _mm256_add_ps(l3, pre_cdf1);

        _mm256_storeu_ps((float*) &lut[1 * size + k], l1);
        _mm256_storeu_ps((float*) &lut[2 * size + k], l2);
        _mm256_storeu_ps((float*) &lut[3 * size + k], l3);
    }

    for (int k = 0; k < size; k += 8) {
        l1 = _mm256_loadu_ps((float const*) &lut[2 * size + k]);
        l2 = _mm256_loadu_ps((float const*) &lut[3 * size + k]);

        l1 = _mm256_mul_ps(l1, m_scale);
        l2 = _mm256_mul_ps(l2, m_scale);

        _mm256_storeu_ps((float*) &lut[2 * size + k], l1);
        _mm256_storeu_ps((float*) &lut[3 * size + k], l2);
    }
}

void cal_lut8(uint32_t *hist, float *lut){
    // find the first non-zero intensity
    int fnz = 0;
    while (!hist[fnz]) ++fnz;

    float scale = (INTENSITY_SPACE - 1.f)/(IMAGE_SIZE - hist[fnz]);
    __m256 m_scale = _mm256_set1_ps(scale);

    // define SIMD variables
    __m256 l1, l2, l3, l4, l5, l6;
    __m256 pre_cdf1, pre_cdf2, pre_cdf3, pre_cdf4;

    // calculate 4 parts of sub-CDF
    int size = INTENSITY_SPACE / 8;
    // #pragma omp parallel num_threads(8)
	for (int id = 0; id < 8; ++id)
	{
        // int id = omp_get_thread_num();
        int start = id * size;
        int end = (id + 1) * size;

        lut[start] = hist[start];
        for (int k = start+1; k < end; ++k) {
            lut[k] = hist[k] + lut[k - 1];
        }
    }

    pre_cdf1 = _mm256_set1_ps(lut[1 * size - 1]);
    pre_cdf2 = _mm256_set1_ps(lut[3 * size - 1]);
    pre_cdf3 = _mm256_set1_ps(lut[5 * size - 1]);
    pre_cdf4 = _mm256_set1_ps(lut[7 * size - 1]);
    for (int k = 0; k < size; k += 8) {
        l1 = _mm256_loadu_ps((float const*) &lut[0 * size + k]);
        l2 = _mm256_loadu_ps((float const*) &lut[1 * size + k]);
        l3 = _mm256_loadu_ps((float const*) &lut[3 * size + k]);
        l4 = _mm256_loadu_ps((float const*) &lut[5 * size + k]);
        l5 = _mm256_loadu_ps((float const*) &lut[7 * size + k]);

        l1 = _mm256_mul_ps(l1, m_scale);
        l2 = _mm256_add_ps(l2, pre_cdf1);
        l3 = _mm256_add_ps(l3, pre_cdf2);
        l4 = _mm256_add_ps(l4, pre_cdf3);
        l5 = _mm256_add_ps(l5, pre_cdf4);

        _mm256_storeu_ps((float*) &lut[0 * size + k], l1);
        _mm256_storeu_ps((float*) &lut[1 * size + k], l2);
        _mm256_storeu_ps((float*) &lut[3 * size + k], l3);
        _mm256_storeu_ps((float*) &lut[5 * size + k], l4);
        _mm256_storeu_ps((float*) &lut[7 * size + k], l5);
    }

    pre_cdf1 = _mm256_set1_ps(lut[2 * size - 1]);
    pre_cdf2 = _mm256_set1_ps(lut[6 * size - 1]);
    for (int k = 0; k < size; k += 8) {
        l1 = _mm256_loadu_ps((float const*) &lut[1 * size + k]);
        l2 = _mm256_loadu_ps((float const*) &lut[2 * size + k]);
        l3 = _mm256_loadu_ps((float const*) &lut[3 * size + k]);
        l4 = _mm256_loadu_ps((float const*) &lut[6 * size + k]);
        l5 = _mm256_loadu_ps((float const*) &lut[7 * size + k]);

        l1 = _mm256_mul_ps(l1, m_scale);
        l2 = _mm256_add_ps(l2, pre_cdf1);
        l3 = _mm256_add_ps(l3, pre_cdf1);
        l4 = _mm256_add_ps(l4, pre_cdf2);
        l5 = _mm256_add_ps(l5, pre_cdf2);

        _mm256_storeu_ps((float*) &lut[1 * size + k], l1);
        _mm256_storeu_ps((float*) &lut[2 * size + k], l2);
        _mm256_storeu_ps((float*) &lut[3 * size + k], l3);
        _mm256_storeu_ps((float*) &lut[6 * size + k], l4);
        _mm256_storeu_ps((float*) &lut[7 * size + k], l5);
    }

    pre_cdf1 = _mm256_set1_ps(lut[4 * size - 1]);
    for (int k = 0; k < size; k += 8) {
        l1 = _mm256_loadu_ps((float const*) &lut[2 * size + k]);
        l2 = _mm256_loadu_ps((float const*) &lut[3 * size + k]);
        l3 = _mm256_loadu_ps((float const*) &lut[4 * size + k]);
        l4 = _mm256_loadu_ps((float const*) &lut[5 * size + k]);
        l5 = _mm256_loadu_ps((float const*) &lut[6 * size + k]);
        l6 = _mm256_loadu_ps((float const*) &lut[7 * size + k]);

        l1 = _mm256_mul_ps(l1, m_scale);
        l2 = _mm256_mul_ps(l2, m_scale);
        l3 = _mm256_add_ps(l3, pre_cdf1);
        l4 = _mm256_add_ps(l4, pre_cdf1);
        l5 = _mm256_add_ps(l5, pre_cdf1);
        l6 = _mm256_add_ps(l6, pre_cdf1);

        _mm256_storeu_ps((float*) &lut[2 * size + k], l1);
        _mm256_storeu_ps((float*) &lut[3 * size + k], l2);
        _mm256_storeu_ps((float*) &lut[4 * size + k], l3);
        _mm256_storeu_ps((float*) &lut[5 * size + k], l4);
        _mm256_storeu_ps((float*) &lut[6 * size + k], l5);
        _mm256_storeu_ps((float*) &lut[7 * size + k], l6);
    }

    for (int k = 0; k < size; k += 8) {
        l1 = _mm256_loadu_ps((float const*) &lut[4 * size + k]);
        l2 = _mm256_loadu_ps((float const*) &lut[5 * size + k]);
        l3 = _mm256_loadu_ps((float const*) &lut[6 * size + k]);
        l4 = _mm256_loadu_ps((float const*) &lut[7 * size + k]);

        l1 = _mm256_mul_ps(l1, m_scale);
        l2 = _mm256_mul_ps(l2, m_scale);
        l3 = _mm256_mul_ps(l3, m_scale);
        l4 = _mm256_mul_ps(l4, m_scale);

        _mm256_storeu_ps((float*) &lut[4 * size + k], l1);
        _mm256_storeu_ps((float*) &lut[5 * size + k], l2);
        _mm256_storeu_ps((float*) &lut[6 * size + k], l3);
        _mm256_storeu_ps((float*) &lut[7 * size + k], l4);
    }
}


uint8_t sat_cast(uint16_t scaled_number){ return (uint8_t)std::min((uint16_t)scaled_number, (uint16_t)0xFFFF); }
