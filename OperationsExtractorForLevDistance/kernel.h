#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cmath>
#include <string>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stack>
#include <chrono>
#define A_SIZE 26
using namespace std;
using namespace std::chrono;

__host__ cudaError_t DistanceMatrixWithCuda(const char* T, const char* P, int* dMatrix, int* xMatrix, const uint32_t tSize, const uint32_t pSize);
__host__ int main(int argc, char** argv);
__host__ void ReadFile(const char* filename, string* line1, string* line2);
__host__ cudaError_t XMatrixWithCuda(const char* T, int* xMatrix, const uint32_t tSize);
__host__ string CalculatePathFromD(int* dMatrix, const char* T, const char* P, const uint32_t tSize, const uint32_t pSize, int* distance);
__host__ void WriteToFile(string result, int distance);
__host__ void WrongArgsPrint();
__host__ void PrintMatrix(int* matrix, uint32_t height, uint32_t width);
__host__ void PrintMatrixToFile(int* matrix, uint32_t height, uint32_t width);