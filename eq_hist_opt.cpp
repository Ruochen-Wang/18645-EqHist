#include <stdlib.h>
#include <stdint.h>
#include <string>
#include <math.h>
#include <iostream>
#include <fstream>
#include <immintrin.h>
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
//void cal_lut(unsigned char *src, uint8_t *lut);
void cal_lut(uint32_t *hist, uint8_t *lut);
uint8_t sat_cast(uint16_t scaled_number);

int main(){
    // prepare input
    std::ifstream input_file("512.b", std::ifstream::binary);
    std::ofstream output_file("512_equalized.b", std::ios::out | std::ios::binary);

    unsigned char *src = new unsigned char[IMAGE_SIZE];
    input_file.read((char *)src, IMAGE_SIZE);

    unsigned char *dst = new unsigned char [IMAGE_SIZE];
	uint32_t *hist = (uint32_t *)malloc(IMAGE_SIZE*sizeof(uint32_t));
	unsigned long long t0 = rdtsc();
//    eq_hist(src, dst);
	unsigned long long t1 = rdtsc();

	printf("delay: %d\n", t1-t0);
	for(int i = 0; i < INTENSITY_SPACE; i++) printf("hist[%d]: %d\n", i, hist[i]);
    output_file.write((char *)dst, IMAGE_SIZE);

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
	// divide into 8 streams, load 32 bytes at once
	for(int i = 0; i < IMAGE_SIZE/8; i += 32){
		a = _mm256_stream_load_si256 ((__m256i const*) (src+i));
		b = _mm256_stream_load_si256 ((__m256i const*) (src+i+8*1024));
		c = _mm256_stream_load_si256 ((__m256i const*) (src+i+16*1024));
		d = _mm256_stream_load_si256 ((__m256i const*) (src+i+24*1024));
		e = _mm256_stream_load_si256 ((__m256i const*) (src+i+32*1024));
		f = _mm256_stream_load_si256 ((__m256i const*) (src+i+40*1024));
		g = _mm256_stream_load_si256 ((__m256i const*) (src+i+48*1024));
		h = _mm256_stream_load_si256 ((__m256i const*) (src+i+56*1024));

		/*TODO: repeat the same extraction for other vectors*/
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

		/*TODO: repeat the same increment for all other 64-bit*/
		hist[a0&0xFF]++;
		hist[(a0>>8)&0xFF]++;
		hist[(a0>>16)&0xFF]++;
		hist[(a0>>24)]++;
		hist[a1&0xFF]++;
		hist[(a1>>8)&0xFF]++;
		hist[(a1>>16)&0xFF]++;
		hist[(a1>>24)]++;
		hist[a2&0xFF]++;
		hist[(a2>>8)&0xFF]++;
		hist[(a2>>16)&0xFF]++;
		hist[(a2>>24)]++;
		hist[a3&0xFF]++;
		hist[(a3>>8)&0xFF]++;
		hist[(a3>>16)&0xFF]++;
		hist[(a3>>24)]++;

		hist[b0&0xFF]++;
		hist[(b0>>8)&0xFF]++;
		hist[(b0>>16)&0xFF]++;
		hist[(b0>>24)]++;
		hist[b1&0xFF]++;
		hist[(b1>>8)&0xFF]++;
		hist[(b1>>16)&0xFF]++;
		hist[(b1>>24)]++;
		hist[b2&0xFF]++;
		hist[(b2>>8)&0xFF]++;
		hist[(b2>>16)&0xFF]++;
		hist[(b2>>24)]++;
		hist[b3&0xFF]++;
		hist[(b3>>8)&0xFF]++;
		hist[(b3>>16)&0xFF]++;
		hist[(b3>>24)]++;

		hist[c0&0xFF]++;
		hist[(c0>>8)&0xFF]++;
		hist[(c0>>16)&0xFF]++;
		hist[(c0>>24)]++;
		hist[c1&0xFF]++;
		hist[(c1>>8)&0xFF]++;
		hist[(c1>>16)&0xFF]++;
		hist[(c1>>24)]++;
		hist[c2&0xFF]++;
		hist[(c2>>8)&0xFF]++;
		hist[(c2>>16)&0xFF]++;
		hist[(c2>>24)]++;
		hist[c3&0xFF]++;
		hist[(c3>>8)&0xFF]++;
		hist[(c3>>16)&0xFF]++;
		hist[(c3>>24)]++;

		hist[d0&0xFF]++;
		hist[(d0>>8)&0xFF]++;
		hist[(d0>>16)&0xFF]++;
		hist[(d0>>24)]++;
		hist[d1&0xFF]++;
		hist[(d1>>8)&0xFF]++;
		hist[(d1>>16)&0xFF]++;
		hist[(d1>>24)]++;
		hist[d2&0xFF]++;
		hist[(d2>>8)&0xFF]++;
		hist[(d2>>16)&0xFF]++;
		hist[(d2>>24)]++;
		hist[d3&0xFF]++;
		hist[(d3>>8)&0xFF]++;
		hist[(d3>>16)&0xFF]++;
		hist[(d3>>24)]++;
	
		a0 = _mm256_extract_epi64(e, 0);
		a1 = _mm256_extract_epi64(e, 1);
		a2 = _mm256_extract_epi64(e, 2);
		a3 = _mm256_extract_epi64(e, 3);
		b0 = _mm256_extract_epi64(f, 0);
		b1 = _mm256_extract_epi64(f, 1);
		b2 = _mm256_extract_epi64(f, 2);
		b3 = _mm256_extract_epi64(f, 3);
		c0 = _mm256_extract_epi64(g, 0);
		c1 = _mm256_extract_epi64(g, 1);
		c2 = _mm256_extract_epi64(g, 2);
		c3 = _mm256_extract_epi64(g, 3);
		d0 = _mm256_extract_epi64(h, 0);
		d1 = _mm256_extract_epi64(h, 1);
		d2 = _mm256_extract_epi64(h, 2);
		d3 = _mm256_extract_epi64(h, 3);

		hist[a0&0xFF]++;
		hist[(a0>>8)&0xFF]++;
		hist[(a0>>16)&0xFF]++;
		hist[(a0>>24)]++;
		hist[a1&0xFF]++;
		hist[(a1>>8)&0xFF]++;
		hist[(a1>>16)&0xFF]++;
		hist[(a1>>24)]++;
		hist[a2&0xFF]++;
		hist[(a2>>8)&0xFF]++;
		hist[(a2>>16)&0xFF]++;
		hist[(a2>>24)]++;
		hist[a3&0xFF]++;
		hist[(a3>>8)&0xFF]++;
		hist[(a3>>16)&0xFF]++;
		hist[(a3>>24)]++;

		hist[b0&0xFF]++;
		hist[(b0>>8)&0xFF]++;
		hist[(b0>>16)&0xFF]++;
		hist[(b0>>24)]++;
		hist[b1&0xFF]++;
		hist[(b1>>8)&0xFF]++;
		hist[(b1>>16)&0xFF]++;
		hist[(b1>>24)]++;
		hist[b2&0xFF]++;
		hist[(b2>>8)&0xFF]++;
		hist[(b2>>16)&0xFF]++;
		hist[(b2>>24)]++;
		hist[b3&0xFF]++;
		hist[(b3>>8)&0xFF]++;
		hist[(b3>>16)&0xFF]++;
		hist[(b3>>24)]++;

		hist[c0&0xFF]++;
		hist[(c0>>8)&0xFF]++;
		hist[(c0>>16)&0xFF]++;
		hist[(c0>>24)]++;
		hist[c1&0xFF]++;
		hist[(c1>>8)&0xFF]++;
		hist[(c1>>16)&0xFF]++;
		hist[(c1>>24)]++;
		hist[c2&0xFF]++;
		hist[(c2>>8)&0xFF]++;
		hist[(c2>>16)&0xFF]++;
		hist[(c2>>24)]++;
		hist[c3&0xFF]++;
		hist[(c3>>8)&0xFF]++;
		hist[(c3>>16)&0xFF]++;
		hist[(c3>>24)]++;

		hist[d0&0xFF]++;
		hist[(d0>>8)&0xFF]++;
		hist[(d0>>16)&0xFF]++;
		hist[(d0>>24)]++;
		hist[d1&0xFF]++;
		hist[(d1>>8)&0xFF]++;
		hist[(d1>>16)&0xFF]++;
		hist[(d1>>24)]++;
		hist[d2&0xFF]++;
		hist[(d2>>8)&0xFF]++;
		hist[(d2>>16)&0xFF]++;
		hist[(d2>>24)]++;
		hist[d3&0xFF]++;
		hist[(d3>>8)&0xFF]++;
		hist[(d3>>16)&0xFF]++;
		hist[(d3>>24)]++;
	

	}
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
