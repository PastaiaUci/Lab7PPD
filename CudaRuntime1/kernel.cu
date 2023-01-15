#include "opencv2/core/core.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include <stdio.h>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
#include <ctime>

const int BLOCKDIM = 32;
const int sigma1 = 50;
const int sigma2 = 50;

__device__ const int FILTER_SIZE = 9;
__device__ const int FILTER_HALFSIZE = FILTER_SIZE >> 1;

using namespace std;


__device__ float exp(int i) { return exp((float)i); }

__global__ void bilateral_filter_2d_unoptimized(unsigned char* input, unsigned char* output, int width, int height, int channel)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x >= FILTER_HALFSIZE) && (x < (width - FILTER_HALFSIZE)) &&
		(y >= FILTER_HALFSIZE) && (y < (height - FILTER_HALFSIZE)))
	{


		for (int c = 0; c < channel; c++) {
			float running_total = 0;
			float norm_factor = 0;
			const int offset = y * width + x;
			for (int xctr = -FILTER_HALFSIZE; xctr <= FILTER_HALFSIZE; xctr++)
			{

				for (int yctr = -FILTER_HALFSIZE; yctr <= FILTER_HALFSIZE; yctr++)
				{
					int y_iter = y + xctr;
					int x_iter = x + yctr;

					float intensity_change = input[(y_iter * width + x_iter) * channel + c] - input[(y * width + x) * channel + c];
					float v1 = exp(-(xctr * xctr + yctr * yctr) / (2 * sigma1 * sigma1));
					float v2 = exp(-(intensity_change * intensity_change) / (2 * sigma2 * sigma2));
					running_total += input[(y_iter * width + x_iter) * channel + c] * v1 * v2;
					norm_factor += v1 * v2;

				}
			}
			output[offset * channel + c] = running_total / norm_factor;
		}
	}
}


void bilateral_filter_wrapper(const cv::Mat& input, cv::Mat& output)
{
	unsigned char* d_input, * d_output;
	cudaError_t cudaStatus;

	int channels = input.step / input.cols;

	cudaStatus = cudaMalloc<unsigned char>(&d_input, input.rows * input.cols * channels);
	checkCudaErrors(cudaStatus);
	cudaStatus = cudaMalloc<unsigned char>(&d_output, output.rows * output.cols * channels);
	checkCudaErrors(cudaStatus);

	cudaStatus = cudaMemcpy(d_input, input.ptr(), input.rows * input.cols * channels, cudaMemcpyHostToDevice);
	checkCudaErrors(cudaStatus);

	const dim3 block(BLOCKDIM, BLOCKDIM);
	const dim3 grid((output.cols + block.x - 1) / block.x, (output.rows + block.y - 1) / block.y);



	bilateral_filter_2d_unoptimized << <grid, block >> > (d_input, d_output, input.cols, input.rows, channels);

	cudaStatus = cudaDeviceSynchronize();
	checkCudaErrors(cudaStatus);

	cudaStatus = cudaMemcpy(output.ptr(), d_output, output.rows * output.cols * channels, cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaStatus);

	cudaStatus = cudaFree(d_input);
	checkCudaErrors(cudaStatus);
	cudaStatus = cudaFree(d_output);
	checkCudaErrors(cudaStatus);
}

int main(int argc, char** argv)
{
	// Read input file (image)
	std::string imagePath = "data/rmn.jpg";
	cv::Mat input = cv::imread(imagePath, cv::ImreadModes::IMREAD_UNCHANGED);
	if (input.empty()) {
		std::cout << "Could not load image. Check location and try again." << std::endl;
		std::cin.get();
		return -1;
	}


	double running_sum = 0.0;
	int attempts = 10;

	cv::Mat output_gpu(input.size(), input.type());
	cv::Mat output_cpu(input.rows, input.cols, CV_8UC1);



	// ------------- BILATERAL FILTER --------------
	int type = atoi(argv[1]);
	if (type == 1) {

		auto begin = std::chrono::high_resolution_clock::now();
		bilateral_filter_wrapper(input, output_gpu);
		auto end = std::chrono::high_resolution_clock::now();
		cv::imwrite("gpu_bilateral_result.png", output_gpu);
		cout << std::chrono::duration_cast <std::chrono::milliseconds>(end - begin).count();
	}
	else {

		auto begin = std::chrono::high_resolution_clock::now();
		cv::bilateralFilter(input, output_cpu, 9, 50, 50);
		auto end = std::chrono::high_resolution_clock::now();
		cv::imwrite("cpu_bilateral_result.png", output_cpu);
		cout << std::chrono::duration_cast <std::chrono::milliseconds>(end - begin).count();

	}
	cv::waitKey();


	return 0;
}


