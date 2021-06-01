//
//  kernel.cu


#include <iostream>
#include <algorithm>
#include <cmath>
#include "ppm.h"

using namespace std;

/*********** Gray Scale Filter  *********/

/**
 * Converts a given 24bpp image into 8bpp grayscale using the GPU.
 */
__global__
void cuda_grayscale(int width, int height, BYTE *image, BYTE *image_out)
{
	int globalIdx = (gridDim.x * blockIdx.y + blockIdx.x) * (blockDim.x * blockDim.y) + (blockDim.x * threadIdx.y + threadIdx.x);
	image_out[globalIdx] = 0.0722f * image[globalIdx*3] + 0.7152f * image[globalIdx*3+1] + 0.2126 * image[globalIdx*3+2];
}


// 1D Gaussian kernel array values of a fixed size (make sure the number > filter size d)
__constant__ float cGaussian[64];
void cuda_updateGaussian(int r, double sd)
{
	
	float fGaussian[64];
	for (int i = 0; i < 2*r +1 ; i++)
	{
		float x = i - r;
		fGaussian[i] = expf(-(x*x) / (2 * sd*sd));
	}
	
	cout << "cudaMemcpyToSymbol cGaussian Error: " << cudaGetErrorString(cudaMemcpyToSymbol(cGaussian, fGaussian, 64*sizeof(float), 0, cudaMemcpyHostToDevice)) << endl;
	
}
__device__
double cuda_gaussian(float x, double sigma)
{
	return expf(-(powf(x, 2)) / (2 * powf(sigma, 2)));
}

/*********** Bilateral Filter  *********/
// Parallel (GPU) Bilateral filter kernel
__global__ void cuda_bilateral_filter(BYTE* input, BYTE* output,
	int width, int height,
	int r, double sI, double sS)
{
	// Global Index
	int globalIdx = (gridDim.x * blockIdx.y + blockIdx.x) * (blockDim.x * blockDim.y) + (blockDim.x * threadIdx.y + threadIdx.x);
	unsigned char centrePx = input[globalIdx];

	int h = (int) (globalIdx / width);
	int w = globalIdx - (width * h);

	double iFiltered = 0;
	double wP = 0;

	for (int dy = -r; dy <= r; dy++)
	{
		int neighborY = h + dy;
		// falls ausserhalb des Bildes:
		if (neighborY < 0) neighborY = 0;
		else if (neighborY >= height) neighborY = height - 1;
		
		for (int dx = -r; dx <= r; dx++)
		{
			int neighborX = w+dx;
			// falls ausserhalb des Bildes:
			if (neighborX < 0) neighborX = 0;
			else if (neighborX >= width) neighborX = width - 1;

			unsigned char curPx = input[neighborY*width+neighborX];
			
			double w = (cGaussian[dy+r] * cGaussian[dx+r]) * cuda_gaussian(centrePx - curPx, sI);
			
			iFiltered += w * curPx;
			wP += w;
		}
	}
	output[globalIdx] = iFiltered / wP;
}


void gpu_pipeline(const Image & input, Image & output, int r, double sI, double sS)
{
	// Events to calculate gpu run time
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// GPU related variables
	BYTE *d_input = NULL;
	BYTE *d_image_out[2] = {0}; //temporary output buffers on gpu device
	int image_size = input.cols*input.rows;
	int suggested_blockSize;   // The launch configurator returned block size 
	int suggested_minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch

	// ******* Grayscale kernel launch *************

	//Creating the block size for grayscaling kernel
	cout << "cudaOccupancyMaxPotentialBlockSize Error: " << cudaGetErrorString(cudaOccupancyMaxPotentialBlockSize( &suggested_minGridSize, &suggested_blockSize, cuda_grayscale)) << endl;
	cout << "Suggested block size: " << suggested_blockSize << " Suggested min grid size: " << suggested_minGridSize << endl;

	int block_dim_x, block_dim_y;
	block_dim_x = block_dim_y = (int) sqrt(suggested_blockSize); 

	dim3 gray_block(block_dim_x, block_dim_y); // 2 pts

	int actual_blockSize = block_dim_x * block_dim_y;
	int actual_gridSize = round(image_size / (float)actual_blockSize + 0.5);

	int grid_dim_x, grid_dim_y;
	grid_dim_x = grid_dim_y = round(sqrt(actual_gridSize)+0.5);
	dim3 gray_grid(grid_dim_x, grid_dim_y);

	cout << "Actual block size: " << actual_blockSize << " Actual grid size: " << actual_gridSize << endl;
	
	// Allocate the intermediate image buffers for each step
	Image img_out(input.cols, input.rows, 1, "P5");
	for (int i = 0; i < 2; i++)
	{  
		cout << "cudaMallocManaged d_img_out Error: " << cudaGetErrorString(cudaMallocManaged(&d_image_out[i], image_size*sizeof(BYTE))) << endl;
		cout << "cudaMemset d_img_out Error: " << cudaGetErrorString(cudaMemset(d_image_out[i], 0, image_size*sizeof(BYTE))) << endl;
	}

	cout << "cudaMallocManaged d_input Error: " << cudaGetErrorString(cudaMallocManaged((void **)&d_input, image_size*3*sizeof(BYTE))) << endl;
	cout << "cudaMemcpy d_input Error: " << cudaGetErrorString(cudaMemcpy(d_input, input.pixels, image_size*3*sizeof(BYTE), cudaMemcpyHostToDevice)) << endl;

	cudaEventRecord(start, 0); // start timer
	
	// Convert input image to grayscale
	cout << "Call 'cuda_grayscale' with gray_grid " << gray_grid.x << " x " << gray_grid.y << " x " << gray_grid.z << endl << "and gray_block " << gray_block.x << " x " << gray_block.y << " x " << gray_block.z << endl << "Image size: " << image_size << endl;
	cuda_grayscale<<<gray_grid, gray_block>>>(input.cols, input.rows, d_input, d_image_out[0]);

	cudaEventRecord(stop, 0); // stop timer
	cudaEventSynchronize(stop);

        // Calculate and print kernel run time
	cudaEventElapsedTime(&time, start, stop);
	cout << "GPU Grayscaling time: " << time << " (ms)\n";
	cout << "Launched blocks of size " << gray_block.x * gray_block.y << endl;
    
	cout << "cudaMemcpy d_image_out Error: " << cudaGetErrorString(cudaMemcpy(img_out.pixels, d_image_out[0], image_size*sizeof(BYTE), cudaMemcpyDeviceToHost)) << endl;
	savePPM(img_out, "image_gpu_gray.ppm");
	

	// ******* Bilateral filter kernel launch *************
	
	//Creating the block size for grayscaling kernel
	cout << "cudaOccupancyMaxPotentialBlockSize Error: " << cudaGetErrorString(cudaOccupancyMaxPotentialBlockSize( &suggested_minGridSize, &suggested_blockSize, cuda_bilateral_filter)) << endl; 
        cout << "Suggested block size: " << suggested_blockSize << " Suggested min grid size: " << suggested_minGridSize << endl;

        block_dim_x = block_dim_y = (int) sqrt(suggested_blockSize); 

        dim3 bilateral_block(block_dim_x, block_dim_y); // 2 pts

	actual_blockSize = block_dim_x * block_dim_y;
	actual_gridSize = round(image_size / (float)actual_blockSize + 0.5);

	grid_dim_x = grid_dim_y = round(sqrt(actual_gridSize)+0.5);
	dim3 bilateral_grid(grid_dim_x, grid_dim_y);

	cout << "Actual block size: " << actual_blockSize << " Actual grid size: " << actual_gridSize << endl;

        // Create gaussain 1d array
        cuda_updateGaussian(r,sS);

        cudaEventRecord(start, 0); // start timer
	
	cuda_bilateral_filter<<<bilateral_grid, bilateral_block>>>(d_image_out[0], d_image_out[1], input.cols, input.rows, r, sI, sS);

        cudaEventRecord(stop, 0); // stop timer
        cudaEventSynchronize(stop);

        // Calculate and print kernel run time
        cudaEventElapsedTime(&time, start, stop);
        cout << "GPU Bilateral Filter time: " << time << " (ms)\n";
        cout << "Launched blocks of size " << bilateral_block.x * bilateral_block.y << endl;

        // Copy output from device to host
	cout << "cudaMemcpy d_image_out Error: " << cudaGetErrorString(cudaMemcpy(output.pixels, d_image_out[1], image_size*sizeof(BYTE), cudaMemcpyDeviceToHost)) << endl;
	
	
        // ************** Finalization, cleaning up ************

        cudaFree(d_image_out);
	cudaFree(d_input);
}
