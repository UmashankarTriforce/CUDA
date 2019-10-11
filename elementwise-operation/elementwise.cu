#include <cuda.h>
#include <stdio.h>

__global__ void vectorAddKernel(double *a, double *b, double *c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        c[i] = a[i] + b[i];
    }
}

__global__ void vectorSubKernel(double *a, double *b, double *c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        c[i] = a[i] - b[i];
    }
}

__global__ void vectorMulKernel(double *a, double *b, double *c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        c[i] = a[i] * b[i];
    }
}

extern "C"{
    double *VectorAdd(double *arrA, double *arrB, int n){
        double *h_c;
        double *d_a, *d_b, *d_c;

        cudaMallocHost((void **) &h_c, sizeof(double) * n);
        cudaMalloc((void **) &d_c, sizeof(double) * n);
        cudaMalloc((void **) &d_a, sizeof(double) * n);
        cudaMalloc((void **) &d_b, sizeof(double) * n);

        cudaMemcpy(d_a, arrA, sizeof(double) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, arrB, sizeof(double) * n, cudaMemcpyHostToDevice);

        vectorAddKernel <<< 2, (n+1/2) >>> (d_a, d_b, d_c, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_c, d_c, sizeof(double) * n, cudaMemcpyDeviceToHost);
        cudaFree(d_c);
        cudaFree(d_a);
        cudaFree(d_b);

        return h_c;
    }

    double *VectorSub(double *arrA, double *arrB, int n){
        double *h_c;
        double *d_a, *d_b, *d_c;

        cudaMallocHost((void **) &h_c, sizeof(double) * n);
        cudaMalloc((void **) &d_c, sizeof(double) * n);
        cudaMalloc((void **) &d_a, sizeof(double) * n);
        cudaMalloc((void **) &d_b, sizeof(double) * n);

        cudaMemcpy(d_a, arrA, sizeof(double) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, arrB, sizeof(double) * n, cudaMemcpyHostToDevice);

        vectorSubKernel <<< 2, (n+1/2) >>> (d_a, d_b, d_c, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_c, d_c, sizeof(double) * n, cudaMemcpyDeviceToHost);
        cudaFree(d_c);
        cudaFree(d_a);
        cudaFree(d_b);

        return h_c;
    }

    double *VectorMul(double *arrA, double *arrB, int n){
        double *h_c;
        double *d_a, *d_b, *d_c;

        cudaMallocHost((void **) &h_c, sizeof(double) * n);
        cudaMalloc((void **) &d_c, sizeof(double) * n);
        cudaMalloc((void **) &d_a, sizeof(double) * n);
        cudaMalloc((void **) &d_b, sizeof(double) * n);

        cudaMemcpy(d_a, arrA, sizeof(double) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, arrB, sizeof(double) * n, cudaMemcpyHostToDevice);

        vectorMulKernel <<< 2, (n+1/2) >>> (d_a, d_b, d_c, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_c, d_c, sizeof(double) * n, cudaMemcpyDeviceToHost);
        cudaFree(d_c);
        cudaFree(d_a);
        cudaFree(d_b);

        return h_c;
    }
}