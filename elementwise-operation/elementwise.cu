#include <cuda.h>
#include <stdio.h>

__global__ void vectorAddKernel(float *a, float *b, float *c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        c[i] = a[i] + b[i];
    }
}

__global__ void vectorSubKernel(float *a, float *b, float *c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        c[i] = a[i] - b[i];
    }
}

__global__ void vectorMulKernel(float *a, float *b, float *c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        c[i] = a[i] * b[i];
    }
}

extern "C"{
    float *VectorAdd(float *arrA, float *arrB, int n){
        float *h_c;
        float *d_a, *d_b, *d_c;

        cudaMallocHost((void **) &h_c, sizeof(float) * n);
        cudaMalloc((void **) &d_c, sizeof(float) * n);
        cudaMalloc((void **) &d_a, sizeof(float) * n);
        cudaMalloc((void **) &d_b, sizeof(float) * n);

        cudaMemcpy(d_a, arrA, sizeof(float) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, arrB, sizeof(float) * n, cudaMemcpyHostToDevice);

        vectorAddKernel <<< 2, (n+1/2) >>> (d_a, d_b, d_c, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_c, d_c, sizeof(float) * n, cudaMemcpyDeviceToHost);
        cudaFree(d_c);
        cudaFree(d_a);
        cudaFree(d_b);

        return h_c;
    }

    float *VectorSub(float *arrA, float *arrB, int n){
        float *h_c;
        float *d_a, *d_b, *d_c;

        cudaMallocHost((void **) &h_c, sizeof(float) * n);
        cudaMalloc((void **) &d_c, sizeof(float) * n);
        cudaMalloc((void **) &d_a, sizeof(float) * n);
        cudaMalloc((void **) &d_b, sizeof(float) * n);

        cudaMemcpy(d_a, arrA, sizeof(float) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, arrB, sizeof(float) * n, cudaMemcpyHostToDevice);

        vectorSubKernel <<< 2, (n+1/2) >>> (d_a, d_b, d_c, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_c, d_c, sizeof(float) * n, cudaMemcpyDeviceToHost);
        cudaFree(d_c);
        cudaFree(d_a);
        cudaFree(d_b);

        return h_c;
    }

    float *VectorMul(float *arrA, float *arrB, int n){
        float *h_c;
        float *d_a, *d_b, *d_c;

        cudaMallocHost((void **) &h_c, sizeof(float) * n);
        cudaMalloc((void **) &d_c, sizeof(float) * n);
        cudaMalloc((void **) &d_a, sizeof(float) * n);
        cudaMalloc((void **) &d_b, sizeof(float) * n);

        cudaMemcpy(d_a, arrA, sizeof(float) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, arrB, sizeof(float) * n, cudaMemcpyHostToDevice);

        vectorMulKernel <<< 2, (n+1/2) >>> (d_a, d_b, d_c, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_c, d_c, sizeof(float) * n, cudaMemcpyDeviceToHost);
        cudaFree(d_c);
        cudaFree(d_a);
        cudaFree(d_b);

        return h_c;
    }
}