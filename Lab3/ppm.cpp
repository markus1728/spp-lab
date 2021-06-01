#include "ppm.h"


Image readPPM(const char *filename) 
{ 
    std::ifstream ifs; 
    ifs.open(filename, std::ios::binary); 
    // need to spec. binary mode for Windows users
    Image src; 
    try { 
        if (ifs.fail()) { 
            throw("Can't open input file");
            exit(1);
        } 
        std::string header; 
        int w, h, b; 
        ifs >> header; 
        if (strcmp(header.c_str(), "P6") != 0) throw("Can't read input file"); 
        ifs >> w >> h >> b; 
        src.cols = w; 
        src.rows = h;
        src.chan = 3;
        src.pixels = new BYTE[w * h * 3]; // this is throw an exception if bad_alloc 
        ifs.ignore(256, '\n'); // skip empty lines in necessary until we get to the binary data 
        BYTE pix[3]; // read each pixel one by one and convert bytes to floats 
        for (int i = 0; i < w * h * 3; ++i) { 
            ifs.read(reinterpret_cast<char *>(pix), 1); 
            src.pixels[i] = pix[0];
        } 
        ifs.close(); 
    } 
    catch (const char *err) { 
        fprintf(stderr, "%s\n", err); 
        ifs.close(); 
    } 
 
    return src; 
} 

void savePPM(const Image &img, const char *filename) 
{ 
    if (img.cols == 0 || img.rows == 0) { fprintf(stderr, "Can't save an empty image\n"); return; } 
    std::ofstream ofs; 
    try { 
        ofs.open(filename, std::ios::binary); // need to spec. binary mode for Windows users 
        if (ofs.fail()) throw("Can't open output file"); 
        ofs << img.ppm_type << "\n" << img.cols << " " << img.rows << "\n255\n"; 
        // loop over each pixel in the image, clamp and convert to byte format
        for (int i = 0; i < img.cols * img.rows * img.chan; ++i) { 
            ofs << img.pixels[i]; 
        } 
        ofs.close(); 
    } 
    catch (const char *err) { 
        fprintf(stderr, "%s\n", err); 
        ofs.close(); 
    } 
}