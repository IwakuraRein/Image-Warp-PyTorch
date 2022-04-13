#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

void ImageWarpKernelLauncher(
	const float* image,
	const float* motion_vector,
	const int* image_size,
	float* output
);

void ImageWarpAccumKernelLauncher(
	const float* image,
	const float* motion_vector,
	const int* image_size,
	const float alpha,
	float* output
);

void ImageWarpGradKernelLauncher(
	const float* image,
	const float* motion_vector,
	const float* backprop,
	const int* image_size,
	float* output
);

void ImageWarpAccumGradKernelLauncher(
	const float* image,
	const float* motion_vector,
	const float* backprop,
	const int* image_size,
	const float alpha,
	float* img_grad_output,
	float* imgdst_grad_output
);