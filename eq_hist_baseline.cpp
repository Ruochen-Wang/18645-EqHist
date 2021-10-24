#include <stdlib.h>
#include <stdint.h>

#define WIDTH                   512
#define HEIGHT                  512
#define IMAGE_SIZE              WID*HGT
#define INTENSITY_SPACE         256

void eq_hist(unsigned char *src, unsigned char *dst);
void cal_lut(unsigned char *src, uint8_t *lut);
void sat_cast(uint16_t scaled_number);

void main(){
    
    unsigned char *dst = malloc(sizeof(unsigned char)*IMAGE_SIZE);
    unsigned char *src = malloc(sizeof(unsigned char)*IMAGE_SIZE);

    // TODO: 1. load input image to src
    eq_hist(src, dst);

    free(src);
    free(dst);
}

void eq_hist(unsigned char *src, unsigned char *dst){
    uint8_t *lut = malloc(sizeof(uint8_t)*INTENSITY_SPACE);
    cal_lut(src, lut);
    // TODO: 1. map; 2. compare
    free(lut);
}

/*
*   @input: src: pointer to an array of IMAGE_SIZE number of unsigned char
*   @output: lut: pointer to an array of INTENSITY_SPACE number of uint8_t
*/
void cal_lut(unsigned char *src, uint8_t *lut){
    int *localHist;
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