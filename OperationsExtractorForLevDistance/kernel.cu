
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t DistanceMatrixWithCuda(char* T, char* P, uint16_t* dMatrix, const int tSize, const int pSize);

__host__ void WrongArgsPrint();

__global__ void CalculateDistanceMatrixKernel(char* T, char* P, uint16_t* dMatrix, int pSize, int tSize)
{
	uint16_t global_tid = threadIdx.x + blockDim.x * blockIdx.x;
	uint8_t lane_id = threadIdx.x % warpSize;
	uint16_t tid = threadIdx.x;
	for (uint16_t row = 0; row <= pSize; row++) {

	}
}

int main(int argc, char** argv)
{
	if (argc != 3) {
		WrongArgsPrint();
		return 1;
	}
	
	cudaError_t cudaStatus = DistanceMatrixWithCuda();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t DistanceMatrixWithCuda(char* T, char* P, uint16_t* dMatrix, const int tSize, const int pSize)
{
	char* dev_T;
	char* dev_P;
	uint16_t* dev_dMatrix;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_T, tSize * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_P, pSize * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_dMatrix, (pSize + 1) * (tSize + 1) * sizeof(uint16_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_T, T, tSize * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_P, P, pSize * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	int threadsPerBlock = 1024;
	int totalThreads = tSize; 
	int blocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
	CalculateDistanceMatrixKernel << <blocks, threadsPerBlock >> > (dev_T, dev_P, dev_dMatrix);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(dMatrix, dev_dMatrix, (pSize + 1) * (tSize + 1) * sizeof(uint16_t), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_P);
	cudaFree(dev_T);
	cudaFree(dev_dMatrix);

	return cudaStatus;
}

__host__ void WrongArgsPrint() {
	printf("Correct way to invoke program is: \"filename s1 s2\"");
}
