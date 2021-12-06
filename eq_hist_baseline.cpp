#include <stdlib.h>
#include <stdint.h>
#include <string>
#include <math.h>
#include <iostream>
#include <fstream>
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
        h1 = H1[i];
        h2 = H2[i];
        result += abs(h1-h2);
    }
    return result/IMAGE_SIZE;
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

    float scale = (INTENSITY_SPACE - 1.f)/(IMAGE_SIZE - localHist[i]);
    int sum = 0;

    // find CDF
    for (lut[i++] = 0; i < INTENSITY_SPACE; ++i)
    {
        sum += localHist[i];
        float scaled_intensity = sum*scale;
        lut[i] = sat_cast((uint16_t) scaled_intensity); // prevent ovf
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
