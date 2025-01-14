#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// compute-sanitizer OperationsExtractorForLevDistance.exe catgactg tactg> a.txt 2>&1

// TODO String Form
#include <stdio.h>
#include <cmath>
#include <string>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stack>
#include <chrono>
#define A_SIZE 26
using namespace std;
using namespace std::chrono;

__host__ cudaError_t DistanceMatrixWithCuda(const char* T, const char* P, int* dMatrix, int* xMatrix, const size_t tSize, const size_t pSize);
__host__ int main(int argc, char** argv);
__host__ void ReadFile(const char* filename, string* line1, string* line2);
__host__ cudaError_t XMatrixWithCuda(const char* T, int* xMatrix, const size_t tSize);
__host__ string CalculatePathFromD(int* dMatrix, const char* T, const char* P, const size_t tSize, const size_t pSize, int* distance);
__host__ void WriteToFile(string result, int distance);
__host__ void WrongArgsPrint();
__host__ void PrintMatrix(int* matrix, size_t height, size_t width);
__host__ void PrintMatrixToFile(int* matrix, size_t height, size_t width);

__global__ void CalculateXMatrixKernel(char* T, int* xMatrix, size_t* tSize) {
	int global_tid = threadIdx.x + blockDim.x * blockIdx.x;
	uint8_t aSize = 26;
	int firstInRow = global_tid * (*tSize + 1);
	xMatrix[firstInRow] = 0;
	for (int i = 1; i <= *tSize; ++i) {
		;		if (T[i] == global_tid + 'A') {
			xMatrix[i + firstInRow] = i;
		}
		else {
			xMatrix[i + firstInRow] = xMatrix[i + firstInRow - 1];
		}
	}
}

__global__ void CalculateDistanceMatrixKernel(char* T, char* P, int* xMatrix, int* dMatrix, size_t* pSize, size_t* tSize)
{
	size_t pSizeLocal = *pSize;
	size_t tSizeLocal = *tSize;
	int global_tid = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;
	uint8_t lane_id = threadIdx.x % 32;
	int warpId = threadIdx.x / 32;

	extern __shared__ char p[];
	char t = T[global_tid];
	int Dvar = 0;
	int Bvar = 0;
	int Cvar = 0;
	int Avar = 0;
	int Xvar = 0;

	for (int i = tid; i <= pSizeLocal; i += blockDim.x) {
		p[i] = P[i];
	}
	if (global_tid > tSizeLocal) return;

	for (int row = 0; row <= pSizeLocal; ++row) {

		__syncthreads();
		Avar = __shfl_up(Dvar, 1);
		if (lane_id == 0 && global_tid != 0) { // waiting for other blocks
			while (dMatrix[(row - 1) * (tSizeLocal + 1) + (global_tid - 1)] == -1) {}
			Avar = dMatrix[(row - 1) * (tSizeLocal + 1) + (global_tid - 1)];
		}

		Bvar = Dvar;

		if (row == 0) {
			Dvar = global_tid;
		}
		else if (global_tid == 0) {
			Dvar = row;
		}
		else if (t == p[row]) {
			Dvar = Avar;
		}
		else if (xMatrix[(p[row] - 'A') * (tSizeLocal + 1) + global_tid] == 0) {
			Dvar = 1 + min(Avar, min(Bvar, row + global_tid - 1));

		}
		else {
			Xvar = xMatrix[(p[row] - 'A') * (tSizeLocal + 1) + global_tid];
			Cvar = dMatrix[(row - 1) * (tSizeLocal + 1) + (Xvar - 1)];
			Dvar = 1 + min(Avar, min(Bvar, Cvar + global_tid - 1 - Xvar));
		}

		dMatrix[row * (tSizeLocal + 1) + global_tid] = Dvar;
	}
}

int main(int argc, char** argv)
{
	if (argc > 2) {
		WrongArgsPrint();
		return 1;
	}
	const char* filename = "dane.txt";
	if (argc == 2)
		filename = argv[1];


	// Read File
	string line1, line2;
	ReadFile(filename, &line1, &line2);
	if (line1 == "" || line2 == "") {
		cerr << "Error reading file" << endl;
		return 1;
	}

	const char* T = line1.c_str();
	const char* P = line2.c_str();

	const size_t tSize = line1.length();
	const size_t pSize = line2.length();
	int distance;

	int* dMatrix = (int*)malloc((tSize + 1) * (pSize + 1) * sizeof(int));
	if (dMatrix == NULL) {
		perror("Memory allocation failed");
		return 1;
	}
	for (size_t i = 0; i < pSize + 1; ++i) {
		for (size_t j = 0; j < tSize + 1; ++j) {
			dMatrix[i * (tSize + 1) + j] = -1;
		}
	}
	int* xMatrix = (int*)malloc((tSize + 1) * 26 * sizeof(int));

	cudaError_t cudaStatus = XMatrixWithCuda(T, xMatrix, tSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "XMatrixWithCuda failed!");
		return 1;
	}
	//PrintMatrix(xMatrix, 26, tSize + 1);


	cudaStatus = DistanceMatrixWithCuda(T, P, dMatrix, xMatrix, tSize, pSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "DistanceMatrixWithCuda failed!");
		return 1;
	}
	cout << "D Calculated" << endl;

	//PrintMatrix(dMatrix, pSize + 1, tSize + 1);
	//PrintMatrixToFile(dMatrix, pSize + 1, tSize + 1);

	string result = CalculatePathFromD(dMatrix, T, P, tSize, pSize, &distance);
	std::cout << result << endl;
	std::cout << distance << endl;
	std::cout << "Distance: " << dMatrix[pSize * (tSize + 1) + tSize] << endl;

	WriteToFile(result, distance);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

__host__ cudaError_t XMatrixWithCuda(const char* T, int* xMatrix, const size_t tSize)
{
	std::chrono::time_point<std::chrono::high_resolution_clock> ts;
	std::chrono::time_point<std::chrono::high_resolution_clock> te;

	char* dev_T;
	size_t* dev_tSize;
	int* dev_xMatrix;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_T, (tSize + 1) * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_xMatrix, A_SIZE * (tSize + 1) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_tSize, sizeof(size_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	// Copy host to device
	cudaStatus = cudaMemcpy(dev_T + 1, T, tSize * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_tSize, &tSize, sizeof(size_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	ts = high_resolution_clock::now();
	CalculateXMatrixKernel << <1, 26 >> > (dev_T, dev_xMatrix, dev_tSize);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	te = high_resolution_clock::now();
	cout << "Time of CalculateDistanceMatrixKernel:    " << setw(7) << 0.001 * duration_cast<microseconds>(te - ts).count() << " nsec" << endl;

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(xMatrix, dev_xMatrix, A_SIZE * (tSize + 1) * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_T);
	cudaFree(dev_xMatrix);
	cudaFree(dev_tSize);

	return cudaStatus;
}

__host__ cudaError_t DistanceMatrixWithCuda(const char* T, const char* P, int* dMatrix, int* xMatrix, const size_t tSize, const size_t pSize)
{
	std::chrono::time_point<std::chrono::high_resolution_clock> ts;
	std::chrono::time_point<std::chrono::high_resolution_clock> te;

	char* dev_T;
	char* dev_P;
	int* dev_dMatrix;
	size_t* dev_tSize;
	size_t* dev_pSize;
	cudaError_t cudaStatus;
	int* dev_xMatrix;
	int threadsPerBlock = 1024;
	size_t totalThreads = tSize + 1;
	int blocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
	cout << "Blocks: " << blocks << " Threads: " << totalThreads << endl;
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_T, (tSize + 1) * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_P, (pSize + 1) * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_dMatrix, (pSize + 1) * (tSize + 1) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_xMatrix, A_SIZE * (tSize + 1) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_tSize, sizeof(size_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_pSize, sizeof(size_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_T + 1, T, tSize * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_P + 1, P, pSize * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_tSize, &tSize, sizeof(size_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_pSize, &pSize, sizeof(size_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_xMatrix, xMatrix, A_SIZE * (tSize + 1) * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_dMatrix, dMatrix, (pSize + 1) * (tSize + 1) * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	ts = high_resolution_clock::now();
	CalculateDistanceMatrixKernel << <blocks, threadsPerBlock, pSize + 1 >> > (dev_T, dev_P, dev_xMatrix, dev_dMatrix, dev_pSize, dev_tSize);

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
	te = high_resolution_clock::now();
	cout << "Time of CalculateDistanceMatrixKernel:    " << setw(7) << 0.001 * duration_cast<microseconds>(te - ts).count() << " nsec" << endl;

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(dMatrix, dev_dMatrix, (pSize + 1) * (tSize + 1) * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_P);
	cudaFree(dev_T);
	cudaFree(dev_dMatrix);
	cudaFree(dev_xMatrix);
	cudaFree(dev_tSize);
	cudaFree(dev_pSize);

	return cudaStatus;
}

__host__ string CalculatePathFromD(int* dMatrix, const char* T, const char* P, const size_t tSize, const size_t pSize, int* distance)
{
	size_t i = pSize;
	size_t j = tSize;
	string result;
	stack<string> path;

	*distance = 0;
	while (i != 0 || j != 0) {
		int minValue = 0;
		(*distance)++;
		if (i != 0 && j != 0)
			minValue = min(dMatrix[(i - 1) * (tSize + 1) + j], min(dMatrix[i * (tSize + 1) + (j - 1)], dMatrix[(i - 1) * (tSize + 1) + (j - 1)]));
		else if (j != 0)
			minValue = dMatrix[i * (tSize + 1) + (j - 1)];
		else
			minValue = dMatrix[(i - 1) * (tSize + 1) + j];
		if (i != 0 && j != 0 && minValue == dMatrix[(i - 1) * (tSize + 1) + (j - 1)]) {
			if (T[j - 1] == P[i - 1]) {
				//path.push("NO OPERATION\n");
				(*distance)--;
			}
			else {
				path.push(string("R, ") + to_string(j - 1) + ", " + P[i - 1] + "\n");
			}
			i--;
			j--;
		}
		else if (j != 0 && minValue == dMatrix[i * (tSize + 1) + (j - 1)]) {
			path.push(string("D, ") + to_string(j - 1) + ", " + T[j - 1] + "\n");
			j--;
		}
		else if (i != 0 && minValue == dMatrix[(i - 1) * (tSize + 1) + j]) {
			path.push(string("I, ") + to_string(j - 1) + ", " + P[i - 1] + "\n");
			i--;
		}
	}
	while (!path.empty()) {
		result.append(path.top());
		path.pop();
	}
	return result;
}
__host__ void WrongArgsPrint() {
	printf("Correct way to invoke program is: \"filename s1 s2\"");
}
__host__ void WriteToFile(string result, int distance) {
	ofstream myfile;
	myfile.open("result.txt");
	myfile << result << endl;
	myfile << "Distance: " << distance;
	myfile.close();
}
__host__ void ReadFile(const char* filename, string* line1, string* line2)
{
	ifstream file(filename);

	if (!file) {
		cerr << "Error opening file: " << filename << endl;
		return;
	}


	if (!getline(file, *line1))
		cerr << "Error reading first line or file is empty." << endl;
	if (!getline(file, *line2))
		cerr << "Error reading second line or file does not have a second line." << endl;
	file.close();
}
__host__ void PrintMatrix(int* matrix, size_t height, size_t width) {
	for (int j = 0; j < height; ++j) {
		for (int i = 0; i < width; ++i) {
			cout << setw(3) << matrix[j * width + i];
		}
		cout << endl;
	}
	cout << endl;
}
__host__ void PrintMatrixToFile(int* matrix, size_t height, size_t width) {
	ofstream myfile;
	myfile.open("DMatrix.txt");
	for (int j = 0; j < height; ++j) {
		for (int i = 0; i < width; ++i) {
			myfile << matrix[j * width + i];
		}
	}
	myfile.close();
}