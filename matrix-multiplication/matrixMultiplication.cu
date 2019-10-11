#include <cublas_v2.h>

extern "C"{
    double *DGEMM(double *A, double *B, int rowA, int colA, int rowB, int colB){
        double *devA, *devB, *devC, *hostC;

        cudaMalloc((void **) &devA, sizeof(double) * (rowA * colA));
        cudaMalloc((void **) &devB, sizeof(double) * (rowB * colB));
        cudaMalloc((void **) &devC, sizeof(double) * (rowA * colB));
        cudaMallocHost((void **) &hostC, sizeof(double) * (rowA * colB));
        cudaMemcpy(devA, A, sizeof(double) * (rowA * colA), cudaMemcpyHostToDevice);
        cudaMemcpy(devB, B, sizeof(double) * (rowB * colB), cudaMemcpyHostToDevice);

        cublasHandle_t handle;
        cublasStatus_t status = cublasCreate(&handle);

        double alpha = 1.0f;double beta = 0.0f;
        status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rowA, colB,\
            colA, &alpha, devA, colA, devB, colB, &beta,  devC, rowA);
        status = cublasDestroy(handle);
        cudaMemcpy(hostC, devC, sizeof(double) * (rowA * colB), cudaMemcpyDeviceToHost);

        cudaFree(devA);
        cudaFree(devB);
        cudaFree(devC);

        return hostC;
    }
}