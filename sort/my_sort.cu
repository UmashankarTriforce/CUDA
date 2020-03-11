#include "my_sort.cuh"

__global__ void _CUDAsort(int* dev, long long int n, int offset) {

	long long int thread = threadIdx.x + blockIdx.x * blockDim.x;

	if (dev[2 * thread + offset] < dev[2 * thread + 1 + offset]) {
		dev[2 * thread + offset] += dev[2 * thread + 1 + offset];
		dev[2 * thread + 1 + offset] = dev[2 * thread + offset] - dev[2 * thread + 1 + offset];
		dev[2 * thread + offset] -= dev[2 * thread + 1 + offset];
	}
}

__host__ void sort(int* host, long long int n) {
	
	int* dev;
	cudaMalloc(&dev, n * sizeof(int));
	cudaMemcpy(dev, host, n * sizeof(int), cudaMemcpyHostToDevice);

	for (long long int i = 0; i < n ; ++i) {
		_CUDAsort <<< n / 2048 + 1, 1024 >>> (dev, n, i % 2);
	}

	cudaMemcpy(host, dev, n * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev);
}
