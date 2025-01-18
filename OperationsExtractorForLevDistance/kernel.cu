#include "kernel.h"
// Kod "rozgrzewający" kartę https://stackoverflow.com/questions/59815212/best-way-to-warm-up-the-gpu-with-cuda (Warm up the GPU)
__global__ void warm_up_gpu() {
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float ia, ib;
	ia = ib = 0.0f;
	ib += ia + tid;
}
__global__ void CalculateXMatrixKernel(char* T, int* xMatrix, uint32_t* tSize) {
	uint32_t global_tid = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t tSizeLocal = *tSize;
	uint32_t firstInRow = global_tid * (tSizeLocal + 1);
	xMatrix[firstInRow] = 0; // Wypełnianie pierwszej kolumny (Filing first column)
	char myLetter = global_tid + 'A';
	for (uint32_t i = 1; i <= tSizeLocal; ++i) {
		if (T[i] == myLetter) {
			xMatrix[i + firstInRow] = i;
		}
		else {
			xMatrix[i + firstInRow] = xMatrix[i + firstInRow - 1];
		}
	}
}

__global__ void CalculateDistanceMatrixKernel(char* T, char* P, int* xMatrix, int* dMatrix, uint32_t* pSize, uint32_t* tSize) {
#pragma region Variables
	uint32_t pSizeLocal = *pSize;
	uint32_t tSizeLocal = *tSize;
	uint32_t global_tid = threadIdx.x + blockDim.x * blockIdx.x;
	uint16_t tid = threadIdx.x;
	uint8_t lane_id = threadIdx.x % 32;
	extern __shared__ char p[]; // Pamięć współdzielona dla p (Shared memory for p)
	char t = T[global_tid]; // Litera z T odpowiadająca wątkowi (Letter from T corresponding to thread)
	int Dvar = 0;
	int Bvar = 0;
	int Cvar = 0;
	int Avar = 0;
	int Xvar = 0;
	int prevDvarId;
	int firstInRowId;
#pragma endregion

	for (int i = tid; i <= pSizeLocal; i += blockDim.x) { // Kopiowanie p do pamięci współdzielonej (Copying p into shared memory)
		p[i] = P[i];
	}

	if (global_tid > tSizeLocal) return; // Kończenie nadmiarowych wątków (returning from redundant threads)

	dMatrix[global_tid] = global_tid; // Wypełnianie pierwszego wiersza (filling first row)
	Dvar = global_tid;

	firstInRowId = 0; // Zmienna pozwalająca uniknięcia operacji mnożenia w pętli (variable avoiding multiplication in loop)

	for (int row = 1; row <= pSizeLocal; ++row) {
		prevDvarId = firstInRowId + (global_tid - 1);

		if (blockIdx.x != 0 && tid == 0)
			while (dMatrix[prevDvarId] == -1) {} // Czekanie na poprzedni blok (waiting for previous block)
		__syncthreads();

		Avar = __shfl_up(Dvar, 1);
		if (lane_id == 0 && global_tid != 0) {
			Avar = dMatrix[prevDvarId];
		}

		if (global_tid == 0) {
			Dvar = row;
		}
		else if (t == p[row]) {
			Dvar = Avar;
		}
		else {
			Xvar = xMatrix[(p[row] - 'A') * (tSizeLocal + 1) + global_tid];
			Bvar = Dvar;
			if (Xvar == 0) {
				Dvar = 1 + min(Avar, min(Bvar, row + global_tid - 1));

			}
			else {
				Cvar = dMatrix[firstInRowId + (Xvar - 1)];
				Dvar = 1 + min(Avar, min(Bvar, Cvar + global_tid - 1 - Xvar));
			}
		}

		firstInRowId += (tSizeLocal + 1);
		dMatrix[firstInRowId + global_tid] = Dvar;
	}
}

int main(int argc, char** argv)
{
	std::chrono::time_point<std::chrono::high_resolution_clock> ts;
	std::chrono::time_point<std::chrono::high_resolution_clock> te;
	ts = high_resolution_clock::now();
#pragma region Variables
	const char* filename = "dane.txt";
	string line1, line2;
	int distance;

#pragma endregion

#pragma region Arguments
	if (argc > 2) {
		WrongArgsPrint();
		return 1;
	}
	if (argc == 2)
		filename = argv[1];
#pragma endregion

	ReadFile(filename, &line1, &line2);
	if (line1 == "" || line2 == "") {
		cerr << "Error reading file" << endl;
		return 1;
	}

	const char* T = line1.c_str();
	const char* P = line2.c_str();
	const uint32_t tSize = line1.length();
	const uint32_t pSize = line2.length();

#pragma region Malloc
	int* dMatrix = (int*)malloc((tSize + 1) * (pSize + 1) * sizeof(int));
	if (dMatrix == NULL) {
		perror("Memory allocation failed for dMatrix");
		return 1;
	}
	int* xMatrix = (int*)malloc((tSize + 1) * 26 * sizeof(int));
	if (dMatrix == NULL) {
		perror("Memory allocation failed for xMatrix");
		return 1;
	}
#pragma endregion

	for (uint32_t i = 0; i < pSize + 1; ++i) { // Ustawianie wartownika (Setting Sentinel)
		for (uint32_t j = 0; j < tSize + 1; ++j) {
			dMatrix[i * (tSize + 1) + j] = -1;
		}
	}

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

	//PrintMatrix(dMatrix, pSize + 1, tSize + 1);
	//PrintMatrixToFile(dMatrix, pSize + 1, tSize + 1);

	string result = CalculatePathFromD(dMatrix, T, P, tSize, pSize, &distance);


	//std::cout << result << endl;
	std::cout << "Distance: " << distance << endl;
	//std::cout << "Distance: " << dMatrix[pSize * (tSize + 1) + tSize] << endl;


	WriteToFile(result, distance);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	free(dMatrix);
	free(xMatrix);
	te = high_resolution_clock::now();
	cout << "Time of whole program:    " << setw(7) << 0.001 * duration_cast<microseconds>(te - ts).count() << " nsec" << endl;
	return 0;
}

__host__ cudaError_t XMatrixWithCuda(const char* T, int* xMatrix, const uint32_t tSize)
{
#pragma region Variables
	std::chrono::time_point<std::chrono::high_resolution_clock> ts;
	std::chrono::time_point<std::chrono::high_resolution_clock> te;
	char* dev_T;
	uint32_t* dev_tSize;
	int* dev_xMatrix;
	cudaError_t cudaStatus;
#pragma endregion

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
#pragma region Malloc

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
	cudaStatus = cudaMalloc((void**)&dev_tSize, sizeof(uint32_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
#pragma endregion
#pragma region Memcpy
	// Copy host to device
	cudaStatus = cudaMemcpy(dev_T + 1, T, tSize * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_tSize, &tSize, sizeof(uint32_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
#pragma endregion
#pragma region WarmUp
	warm_up_gpu << <1, 26 >> > ();
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "warm_up_gpu launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching warm_up_gpu!\n", cudaStatus);
		goto Error;
	}
#pragma endregion
	// Launch a kernel on the GPU with one thread for each element.
	ts = high_resolution_clock::now();
	CalculateXMatrixKernel << <1, 26 >> > (dev_T, dev_xMatrix, dev_tSize);
#pragma region PostKernel
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
#pragma endregion

	Error:
	cudaFree(dev_T);
	cudaFree(dev_xMatrix);
	cudaFree(dev_tSize);

	return cudaStatus;
}

__host__ cudaError_t DistanceMatrixWithCuda(const char* T, const char* P, int* dMatrix, int* xMatrix, const uint32_t tSize, const uint32_t pSize)
{
#pragma region Variables
	std::chrono::time_point<std::chrono::high_resolution_clock> ts;
	std::chrono::time_point<std::chrono::high_resolution_clock> te;
	char* dev_T;
	char* dev_P;
	int* dev_dMatrix;
	uint32_t* dev_tSize;
	uint32_t* dev_pSize;
	cudaError_t cudaStatus;
	int* dev_xMatrix;
	int threadsPerBlock = 1024;
	uint32_t totalThreads = tSize + 1;
	int blocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
#pragma endregion

	cout << "Blocks: " << blocks << " Threads: " << totalThreads << endl;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
#pragma region Malloc
	// Allocate GPU buffers for three vectors (two input, one output).
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
	cudaStatus = cudaMalloc((void**)&dev_tSize, sizeof(uint32_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_pSize, sizeof(uint32_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
#pragma endregion

#pragma region Memcpy
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
	cudaStatus = cudaMemcpy(dev_tSize, &tSize, sizeof(uint32_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_pSize, &pSize, sizeof(uint32_t), cudaMemcpyHostToDevice);
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
#pragma endregion
#pragma region WarmUp
	warm_up_gpu << <blocks, threadsPerBlock >> > ();
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "warm_up_gpu launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching warm_up_gpu!\n", cudaStatus);
		goto Error;
	}
	// Launch a kernel on the GPU with one thread for each element.
#pragma endregion
	ts = high_resolution_clock::now();
	CalculateDistanceMatrixKernel << <blocks, threadsPerBlock, pSize + 1 >> > (dev_T, dev_P, dev_xMatrix, dev_dMatrix, dev_pSize, dev_tSize);
#pragma region PostKernel

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
	cudaStatus = cudaMemcpy(dMatrix, dev_dMatrix, (pSize + 1) * (tSize + 1) * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
#pragma endregion

	Error:
	cudaFree(dev_P);
	cudaFree(dev_T);
	cudaFree(dev_dMatrix);
	cudaFree(dev_xMatrix);
	cudaFree(dev_tSize);
	cudaFree(dev_pSize);

	return cudaStatus;
}

__host__ string CalculatePathFromD(int* dMatrix, const char* T, const char* P, const uint32_t tSize, const uint32_t pSize, int* distance)
{
	uint32_t i = pSize;
	uint32_t j = tSize;
	string result;
	stack<string> path;
	ostringstream operation;

	*distance = 0;
	while (i != 0 || j != 0) {
		int minValue = 0;
		operation.clear();
		operation.str("");
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
				operation << "R, " << i - 1 << ", " << P[i - 1] << "\n";
			}
			i--;
			j--;
		}
		else if (j != 0 && minValue == dMatrix[i * (tSize + 1) + (j - 1)]) {
			operation << "D, " << i << ", " << T[j - 1] << "\n";
			j--;
		}
		else if (i != 0 && minValue == dMatrix[(i - 1) * (tSize + 1) + j]) {
			operation << "I, " << i - 1 << ", " << P[i - 1] << "\n";
			i--;
		}
		path.push(operation.str());
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
__host__ void PrintMatrix(int* matrix, uint32_t height, uint32_t width) {
	for (uint32_t j = 0; j < height; ++j) {
		for (uint32_t i = 0; i < width; ++i) {
			cout << setw(3) << matrix[j * width + i];
		}
		cout << endl;
	}
	cout << endl;
}
__host__ void PrintMatrixToFile(int* matrix, uint32_t height, uint32_t width) {
	ofstream myfile;
	myfile.open("DMatrix.txt");
	for (uint32_t j = 0; j < height; ++j) {
		for (uint32_t i = 0; i < width; ++i) {
			myfile << matrix[j * width + i] << " ";
		}
		myfile << endl;
	}
	myfile.close();
}