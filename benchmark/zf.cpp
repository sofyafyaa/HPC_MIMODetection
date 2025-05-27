#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <chrono>

using namespace std;
using namespace chrono;

// CUDA complex
using Complex = cuFloatComplex;

// Random complex number generator
Complex rand_complex() {
    float re = static_cast<float>(rand()) / RAND_MAX;
    float im = static_cast<float>(rand()) / RAND_MAX;
    return make_cuFloatComplex(re, im);
}

// CUDA error checking (optional but useful)
#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); return EXIT_FAILURE;}} while(0)

#define CUBLAS_CALL(x) do { if((x)!=CUBLAS_STATUS_SUCCESS) { \
    printf("cuBLAS error at %s:%d\n",__FILE__,__LINE__); return EXIT_FAILURE;}} while(0)

#define CUSOLVER_CALL(x) do { if((x)!=CUSOLVER_STATUS_SUCCESS) { \
    printf("cuSolver error at %s:%d\n",__FILE__,__LINE__); return EXIT_FAILURE;}} while(0)


// GPU ZF implementation
float gpu_zf(int rx, int tx, Complex* d_H, Complex* d_y, Complex* d_x,
             cublasHandle_t cublasH, cusolverDnHandle_t solverH) {
    
    Complex *d_HH, *d_HH_inv, *d_Hy;
    CUDA_CALL(cudaMalloc(&d_HH, sizeof(Complex) * tx * tx));
    CUDA_CALL(cudaMalloc(&d_HH_inv, sizeof(Complex) * tx * tx));
    CUDA_CALL(cudaMalloc(&d_Hy, sizeof(Complex) * tx));

    float one = 1.0f;
    float zero = 0.0f;
    Complex cone = make_cuFloatComplex(1.0f, 0.0f);
    Complex czero = make_cuFloatComplex(0.0f, 0.0f);

    auto start = high_resolution_clock::now();

    // H^H * H -> Hermitian matrix (only lower part used)
    CUBLAS_CALL(cublasCherk(cublasH, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_C, 
                            tx, rx, &one, d_H, rx, &zero, d_HH, tx));

    // Cholesky factorization (H^H H = L * L^H)
    int workspace_bytes = 0;
    int* dev_info;
    CUDA_CALL(cudaMalloc(&dev_info, sizeof(int)));

    CUSOLVER_CALL(cusolverDnCpotrf_bufferSize(solverH, CUBLAS_FILL_MODE_LOWER, tx, d_HH, tx, &workspace_bytes));

    Complex* d_work;
    CUDA_CALL(cudaMalloc(&d_work, workspace_bytes * sizeof(Complex)));

    CUSOLVER_CALL(cusolverDnCpotrf(solverH, CUBLAS_FILL_MODE_LOWER, tx, d_HH, tx, d_work, workspace_bytes, dev_info));
    CUSOLVER_CALL(cusolverDnCpotri(solverH, CUBLAS_FILL_MODE_LOWER, tx, d_HH, tx, d_work, workspace_bytes, dev_info));

    CUDA_CALL(cudaMemcpy(d_HH_inv, d_HH, sizeof(Complex) * tx * tx, cudaMemcpyDeviceToDevice));

    // H^H * y
    CUBLAS_CALL(cublasCgemv(cublasH, CUBLAS_OP_C, rx, tx, &cone, d_H, rx, d_y, 1, &czero, d_Hy, 1));

    // x = inv(H^H H) * H^H y
    CUBLAS_CALL(cublasCgemv(cublasH, CUBLAS_OP_N, tx, tx, &cone, d_HH_inv, tx, d_Hy, 1, &czero, d_x, 1));

    CUDA_CALL(cudaDeviceSynchronize());
    auto end = high_resolution_clock::now();
    float seconds = duration_cast<duration<float>>(end - start).count();

    // Cleanup
    CUDA_CALL(cudaFree(d_HH));
    CUDA_CALL(cudaFree(d_HH_inv));
    CUDA_CALL(cudaFree(d_Hy));
    CUDA_CALL(cudaFree(d_work));
    CUDA_CALL(cudaFree(dev_info));

    return seconds;
}

int main() {
    vector<int> tx_sizes = {32, 64, 128, 256};
    const int RX_TX_RATIO = 2;

    cublasHandle_t cublasH;
    cusolverDnHandle_t solverH;
    cublasCreate(&cublasH);
    cusolverDnCreate(&solverH);

    for (int tx : tx_sizes) {
        int rx = tx * RX_TX_RATIO;

        vector<Complex> h_H(rx * tx);
        vector<Complex> h_y(rx);
        vector<Complex> h_x(tx);

        for (auto& v : h_H) v = rand_complex();
        for (auto& v : h_y) v = rand_complex();

        // Simulated CPU (placeholder)
        auto t1 = high_resolution_clock::now();
        h_x = vector<Complex>(tx, make_cuFloatComplex(0, 0)); // dummy
        auto t2 = high_resolution_clock::now();
        float cpu_time = duration_cast<duration<float>>(t2 - t1).count();

        // Allocate device memory
        Complex *d_H, *d_y, *d_x;
        CUDA_CALL(cudaMalloc(&d_H, sizeof(Complex) * rx * tx));
        CUDA_CALL(cudaMalloc(&d_y, sizeof(Complex) * rx));
        CUDA_CALL(cudaMalloc(&d_x, sizeof(Complex) * tx));


        CUDA_CALL(cudaMemcpy(d_H, h_H.data(), sizeof(Complex) * rx * tx, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_y, h_y.data(), sizeof(Complex) * rx, cudaMemcpyHostToDevice));

        float gpu_time = gpu_zf(rx, tx, d_H, d_y, d_x, cublasH, solverH);

        CUDA_CALL(cudaFree(d_H));
        CUDA_CALL(cudaFree(d_y));
        CUDA_CALL(cudaFree(d_x));

        cout << "TX: " << tx << ", RX: " << rx
             << " | CPU: " << cpu_time << " s"
             << " | GPU: " << gpu_time << " s"
             << " | Speedup: " << (cpu_time / gpu_time) << "x\n";
    }

    cublasDestroy(cublasH);
    cusolverDnDestroy(solverH);
    return 0;
}
