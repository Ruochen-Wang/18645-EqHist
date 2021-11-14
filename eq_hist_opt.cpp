#include <stdlib.h>
#include <stdint.h>
#include <string>
#include <math.h>
#include <iostream>
#include <fstream>
#include <immintrin.h>
#include <malloc.h>
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
void singlestream_cal_hist(unsigned char *src, uint32_t *hist);
void base_cal_hist(unsigned char *src, uint32_t *hist);
//void cal_lut(unsigned char *src, uint8_t *lut);
void cal_lut(uint32_t *hist, uint8_t *lut);
uint8_t sat_cast(uint16_t scaled_number);

int main(){
    // prepare input
    std::ifstream input_file("512.b", std::ifstream::binary);
//    std::ofstream output_file("512_equalized.b", std::ios::out | std::ios::binary);

//    unsigned char *src = new unsigned char[IMAGE_SIZE];
	unsigned char *src = (unsigned char *)memalign(256, IMAGE_SIZE*sizeof(uint8_t));
    input_file.read((char *)src, IMAGE_SIZE);

	printf("reading...\n");
	__m256i a = _mm256_stream_load_si256 ((__m256i const*) (src));
	printf("read here succeed\n");
    unsigned char *dst = new unsigned char [IMAGE_SIZE];
	uint32_t *hist1 = (uint32_t *)malloc(INTENSITY_SPACE*sizeof(uint32_t));
	uint32_t *hist2 = (uint32_t *)malloc(INTENSITY_SPACE*sizeof(uint32_t));
	uint32_t *hist3 = (uint32_t *)malloc(INTENSITY_SPACE*sizeof(uint32_t));

	unsigned long long t0, t1;	

	t0 = rdtsc();
	base_cal_hist(src, hist1);
	t1 = rdtsc();
	printf("baseline delay: %d\n", t1-t0);

	t0 = rdtsc();
	singlestream_cal_hist(src, hist2);
	t1 = rdtsc();
	printf("singlestream delay: %d\n", t1-t0);

	t0 = rdtsc();
	multistream_cal_hist(src, hist3);
	t1 = rdtsc();
	printf("multistream delay: %d\n", t1-t0);
/*
	for(int i = 0; i < 32; i+=4){
		printf("src: %d, %d, %d, %d\n", src[i], src[i+1], src[i+2], src[i+3]);
	}
*/
	int hist1tt = 0, hist2tt = 0, hist3tt = 0;
	for(int i = 0; i < INTENSITY_SPACE; i++){
		hist1tt += hist1[i];
		hist3tt += hist3[i];
		if(hist1[i] != hist3[i])
			printf("at %d actually: %d, expected: %d\n", i, hist1[i], hist3[i]);	
	}
	printf("total actually: %d, expected: %d\n", hist1tt, hist3tt);

//	for(int i = 0; i < INTENSITY_SPACE; i++) printf("hist[%d]: %d\n", i, hist[i]);
//    output_file.write((char *)dst, IMAGE_SIZE);
	free(hist1);
	free(hist2);
	free(hist3);
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

void base_cal_hist(unsigned char *src, uint32_t *hist){
    for (int i = 0; i < IMAGE_SIZE; i++)
        hist[(unsigned char)src[i]]++;
}

void cal_lut(uint32_t *hist, uint8_t *lut){
    // find the first non-zero intensity
    int i = 0;
    while (!hist[i]) ++i;

    float scale = (INTENSITY_SPACE - 1.f)/(IMAGE_SIZE - hist[i]);
    int sum = 0;

    // find CDF
    for (lut[i++] = 0; i < INTENSITY_SPACE; ++i)
    {
        sum += hist[i];
        float scaled_intensity = sum*scale;
        lut[i] = sat_cast((uint16_t) scaled_intensity); // prevent ovf
    }
}

uint8_t sat_cast(uint16_t scaled_number){ return (uint8_t)std::min((uint16_t)scaled_number, (uint16_t)0xFFFF); }
