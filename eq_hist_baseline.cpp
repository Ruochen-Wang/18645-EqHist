#include <stdlib.h>
#include <stdint.h>
#include <string>
#include <math.h>
#include "opencv2/imgcodecs.hpp"

using namespace cv;
using namespace std;

#define WIDTH                   512
#define HEIGHT                  512
#define IMAGE_SIZE              WID*HGT
#define INTENSITY_SPACE         256

void eq_hist(unsigned char *src, unsigned char *dst);
void cal_lut(unsigned char *src, uint8_t *lut);
void sat_cast(uint16_t scaled_number);

void main(){
    string src_path = "./input.jpg";
    string dst_path = "./output.jpg"
    Mat src = imread( samples::findFile(src_path), IMREAD_COLOR );
    if( src.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        return -1;
    }

    cvtColor( src, src, COLOR_BGR2GRAY);

    Mat dst;
    //unsigned char *dst = malloc(sizeof(unsigned char)*IMAGE_SIZE);
    eq_hist(src, dst);
    imwrite(dst_path, dst);

    free(src);
    free(dst);
}

void eq_hist(unsigned char *src, unsigned char *dst){
    uint8_t *lut = malloc(sizeof(uint8_t)*INTENSITY_SPACE);
    cal_lut(src, lut);
    
    for (int i = 0; i < IMAGE_SIZE; ++i) {
        dst[i] = lut[src[i]];
    }

    free(lut);
}

void compare_hist(unsigned int *H1, unsigned int *H2){
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
    int *localHist;
    int total = IMAGE_SIZE;
    int hist_sz = INTENSITY_SPACE;

    localHist = calloc(sizeof(int), INTENSITY_SPACE);

    // collect histogram
    for (int i = 0; i < SZ; i)
        localHist[src[i]]++;

    // find the first non-zero intensity
    int i = 0;
    while (!localHist[i]) ++i;

    float scale = (hist_sz - 1.f)/(total - hist[i]);
    int sum = 0;

    // find CDF
    for (lut[i++] = 0; i < INTENSITY_SPACE; ++i)
    {
        sum += localHist[i];
        float scaled_intensity = sum*scale;
        lut[i] = sat_cast((uint16_t) scaled_intensity); // prevent ovf
    }
}

uint8_t sat_cast(uint16_t scaled_number){ return (uint8_t)std::min(scaled_number, 0xFFFF); }
