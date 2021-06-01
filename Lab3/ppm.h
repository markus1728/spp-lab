#ifndef PPM_H
#define PPM_H

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <string.h>
#include <fstream>

using namespace std;

typedef unsigned char BYTE;

class Image 
{ 
public: 
    unsigned int cols, rows, chan; // Image resolution 
    BYTE *pixels; // 1D array of pixels
    string ppm_type;
 
    Image() : cols(0), rows(0), chan(0), pixels(nullptr), ppm_type("P6") { /* empty image */ } 
    Image(const unsigned int &_cols, const unsigned int &_rows, const unsigned int &_chan, const string &type) : 
        cols(_cols), rows(_rows), chan(_chan), pixels(NULL) {
            pixels = new BYTE[rows * cols * chan];
            ppm_type = type;
        }

    const BYTE& operator [] (const unsigned int &i) const {
        return pixels[i];
    } 
    
    BYTE& operator [] (const unsigned int &i) {
        return pixels[i];
    }

    ~Image() {
        if (pixels != NULL)
            delete [] pixels;
    } 
    
}; 

Image readPPM(const char *filename);

void savePPM(const Image &img, const char *filename);

#endif