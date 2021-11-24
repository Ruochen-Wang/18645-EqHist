#include <stdlib.h>
#include <stdint.h>
#include <string>
#include <math.h>
#include <iostream>
#include <fstream>
#include "immintrin.h"
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

void eq_hist(unsigned char *src, unsigned char *dst);
void cal_lut(unsigned char *src, uint8_t *lut);
unsigned int *gen_hist(unsigned char *src, unsigned int *ptr);
float compare_hist(unsigned int *H1, unsigned int *H2);
uint8_t sat_cast(uint16_t scaled_number);


int main(){
    // prepare input
    std::ifstream input_file("512.b", std::ifstream::binary);
    std::ofstream output_file("512_equalized.b", std::ios::out | std::ios::binary);
    std::ifstream input_file0("comp1.b", std::ifstream::binary); //pic1 for comparison
    std::ifstream input_file1("comp2.b", std::ifstream::binary); //pic2 for comparison

    unsigned char *src = new unsigned char[IMAGE_SIZE];
    input_file.read((char *)src, IMAGE_SIZE);
    //compare
    unsigned int* ptr1 = (unsigned int*) malloc (sizeof(unsigned int)*INTENSITY_SPACE);
    unsigned int* ptr2 = (unsigned int*) malloc (sizeof(unsigned int)*INTENSITY_SPACE);
    unsigned int *h1, *h2;
    float result=0;

    unsigned char *src0 = new unsigned char[IMAGE_SIZE];
    input_file0.read((char *)src0, IMAGE_SIZE);
    unsigned char *src1 = new unsigned char[IMAGE_SIZE];
    input_file1.read((char *)src1, IMAGE_SIZE);

    unsigned char *dst = new unsigned char [IMAGE_SIZE];
	unsigned long long t0 = rdtsc();
    eq_hist(src, dst);
    h1 = gen_hist(src0, ptr1);
    h2 = gen_hist(src1, ptr2);
    result = compare_hist(h1, h2);
	unsigned long long t1 = rdtsc();

	printf("delay: %d\n", t1-t0);
    printf("result: %.6f\n", result);
    output_file.write((char *)dst, IMAGE_SIZE);

    return 0;
}


void eq_hist(unsigned char *src, unsigned char *dst){
    uint8_t *lut = new uint8_t[INTENSITY_SPACE];
    cal_lut(src, lut);
    
    for (int i = 0; i < IMAGE_SIZE; ++i) {
        dst[i] = lut[(unsigned char)src[i]];
    }

}

float compare_hist(unsigned int *H1, unsigned int *H2){
    // TODO: 
    //1. normalize H1 and H2 by dividing the IMAGE_SIZE.
    //2. calculate distance of two array H1 and H2. (L1 distance)
    float result = 0;
    float h1, h2;

    for (int i = 0; i < INTENSITY_SPACE; ++i){
        h1 = H1[i] / IMAGE_SIZE;
        h2 = H2[i] / IMAGE_SIZE;
        result += abs(h1-h2);
    }
    return result;
}

/*
*   @input: src: pointer to an array of IMAGE_SIZE number of unsigned char
*   @output: lut: pointer to an array of INTENSITY_SPACE number of uint8_t
*/
void cal_lut(unsigned char *src, uint8_t *lut){
    int localHist[INTENSITY_SPACE] = {};

    // collect histogram
    for (int i = 0; i < IMAGE_SIZE; i++)
        localHist[(unsigned char)src[i]]++;

    // find the first non-zero intensity
    int i = 0;
    while (!localHist[i]) ++i;

    double scale = (INTENSITY_SPACE - 1.f)/(IMAGE_SIZE - localHist[i]);

    // define SIMD variables
    __m256d simd_scale = _mm256_set1_pd(scale);
    __m256d l0, l1, l2, l3;
    __m256d pre_cdf;

    // calculate 4 parts of sub-CDF
    int size = INTENSITY_SPACE / 4;
    for (int n = 0; n < 4; ++n) {
        // parallelizable
        for (int i = 0; i < size; ++i) {
            for (int j = n * size; j < i; ++j) {
                lut[n * size + i] += localHist[n * size + j];
            }
        }
    }

    // 1. normalize cdf of the first part
    // 2. calculate true cdf of the second part
    // 3. update cdf of the fourth part
    pre_cdf = _mm256_broadcast_sd(&lut[size - 1]);
    for (int i = 0; i < size; ++i) {
        l0 = _mm256_load_pd(&lut[i]);
        l1 = _mm256_load_pd(&lut[size + i]);
        l3 = _mm256_load_pd(&lut[3 * size + i]);

        l0 = l0 * simd_scale;
        l1 = l1 + pre_cdf;
        l3 = l3 + pre_cdf;

        lut[i] = _mm256_store_pd(&lut[i], l0);
        lut[size + i] = _mm256_store_pd(&lut[size + i], l1);
        lut[3 * size + i] = _mm256_store_pd(&lut[3 * size + i], l1);
    }

    // 1. normalize cdf of the second part
    // 2. calculate true cdf of the third part
    // 3. update cdf of the fourth part
    pre_cdf = _mm256_broadcast_sd(&lut[2 * size - 1]);
    for (int i = 0; i < size; ++i) {
        l1 = _mm256_load_pd(&lut[size + i]);
        l2 = _mm256_load_pd(&lut[2 * size + i]);
        l3 = _mm256_load_pd(&lut[3 * size + i]);

        l1 = l1 * simd_scale;
        l2 = l2 + pre_cdf;
        l3 = l3 + pre_cdf;

        lut[size + i] = _mm256_store_pd(&lut[size + i], l1);
        lut[2 * size + i] = _mm256_store_pd(&lut[2 * size + i], l2);
        lut[3 * size + i] = _mm256_store_pd(&lut[3 * size + i], l3);
    }

    // 1. normalize cdf of the third part
    // 2. normalize cdf of the fourth part
    for (int i = 0; i < size; ++i) {
        l2 = _mm256_load_pd(&lut[2 * size + i]);
        l3 = _mm256_load_pd(&lut[3 * size + i]);  

        l2 = l2 * simd_scale;
        l3 = l3 * simd_scale;

        lut[2 * size + i] = _mm256_store_pd(&lut[2 * size + i], l2);
        lut[3 * size + i] = _mm256_store_pd(&lut[3 * size + i], l3);
    }
}

unsigned int *gen_hist(unsigned char *src, unsigned int *ptr){
    // collect histogram
    for (int i = 0; i < IMAGE_SIZE; i++){
        ptr[(unsigned char)src[i]]++;
    }
    return ptr;
}

uint8_t sat_cast(uint16_t scaled_number){ return (uint8_t)std::min((uint16_t)scaled_number, (uint16_t)0xFFFF); }
