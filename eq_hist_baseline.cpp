#include <stdlib.h>
#include <stdint.h>
#include <string>
#include <math.h>
#include <iostream>
#include <fstream>
//#include "opencv2/imgcodecs.hpp"

//using namespace cv;
//using namespace std;

#define WIDTH                   512
#define HEIGHT                  512
#define IMAGE_SIZE              WIDTH*HEIGHT
#define INTENSITY_SPACE         256

void eq_hist(unsigned char *src, unsigned char *dst);
void cal_lut(unsigned char *src, uint8_t *lut);
uint8_t sat_cast(uint16_t scaled_number);

int main(){
    // prepare input
    std::ifstream input_file("0a9da1fa077e_d510dfa4b13d.b", std::ifstream::binary);
    std::ofstream output_file("0a9da1fa077e_d510dfa4b13d_equalized.b", std::ios::out | std::ios::binary);

    unsigned char *src = new unsigned char[IMAGE_SIZE];
    input_file.read((char *)src, IMAGE_SIZE);

    unsigned char *dst = new unsigned char [IMAGE_SIZE];
    eq_hist(src, dst);

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

uint8_t sat_cast(uint16_t scaled_number){ return (uint8_t)std::min((uint16_t)scaled_number, (uint16_t)0xFFFF); }
