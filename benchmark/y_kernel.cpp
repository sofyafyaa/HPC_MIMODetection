#include <iostream>
#include <vector>
#include <complex>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <cuComplex.h>

using namespace std;
using namespace chrono;
using Complex = complex<float>;

// ================= CUDA Kernel =================
__global__ void compute_y_kernel(
    cuFloatComplex* H, cuFloatComplex* x,
    cuFloatComplex* noise, cuFloatComplex* y,
    int n_symbols, int n_subcarriers, int n_rx, int n_tx) {

    int s = blockIdx.x;      // symbol index
    int sc = threadIdx.x;    // subcarrier index

    if (sc >= n_subcarriers || s >= n_symbols) return;

    for (int rx = 0; rx < n_rx; ++rx) {
        cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);
        for (int tx = 0; tx < n_tx; ++tx) {
            int h_idx = ((s * n_subcarriers + sc) * n_rx + rx) * n_tx + tx;
            int x_idx = ((s * n_subcarriers + sc) * n_tx) + tx;
            sum = cuCaddf(sum, cuCmulf(H[h_idx], x[x_idx]));
        }
        int y_idx = ((s * n_subcarriers + sc) * n_rx) + rx;
        y[y_idx] = cuCaddf(sum, noise[y_idx]);
    }
}

// ================= CPU Reference =================
void channel_cpu(
    const vector<Complex>& H, const vector<Complex>& x,
    const vector<Complex>& noise, vector<Complex>& y,
    int n_symbols, int n_subcarriers, int n_rx, int n_tx) {

    for (int s = 0; s < n_symbols; ++s) {
        for (int sc = 0; sc < n_subcarriers; ++sc) {
            for (int rx = 0; rx < n_rx; ++rx) {
                Complex sum = 0.0f;
                for (int tx = 0; tx < n_tx; ++tx) {
                    int h_idx = ((s * n_subcarriers + sc) * n_rx + rx) * n_tx + tx;
                    int x_idx = ((s * n_subcarriers + sc) * n_tx) + tx;
                    sum += H[h_idx] * x[x_idx];
                }
                int y_idx = ((s * n_subcarriers + sc) * n_rx) + rx;
                y[y_idx] = sum + noise[y_idx];
            }
        }
    }
}

int main() {
    const int n_symbols = 4;
    const int n_subcarriers = 1024 * 64;  // adjust for size sweep
    const int n_rx = 4, n_tx = 4;

    int H_size = n_symbols * n_subcarriers * n_rx * n_tx;
    int x_size = n_symbols * n_subcarriers * n_tx;
    int y_size = n_symbols * n_subcarriers * n_rx;

    vector<Complex> H(H_size), x(x_size), noise(y_size), y_cpu(y_size), y_gpu(y_size);

    mt19937 rng(42);
    uniform_real_distribution<float> dist(0.f, 1.f);
    for (auto& h : H) h = {dist(rng), dist(rng)};
    for (auto& xx : x) xx = {dist(rng), dist(rng)};
    for (auto& n : noise) n = {dist(rng) * 0.01f, dist(rng) * 0.01f};

    // CPU benchmark
    auto t1 = high_resolution_clock::now();
    channel_cpu(H, x, noise, y_cpu, n_symbols, n_subcarriers, n_rx, n_tx);
    auto t2 = high_resolution_clock::now();
    float cpu_time = duration<float>(t2 - t1).count();

    // Allocate device memory
    cuFloatComplex *d_H, *d_x, *d_noise, *d_y;
    cudaMalloc(&d_H, sizeof(cuFloatComplex) * H_size);
    cudaMalloc(&d_x, sizeof(cuFloatComplex) * x_size);
    cudaMalloc(&d_noise, sizeof(cuFloatComplex) * y_size);
    cudaMalloc(&d_y, sizeof(cuFloatComplex) * y_size);

    cudaMemcpy(d_H, H.data(), sizeof(cuFloatComplex) * H_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), sizeof(cuFloatComplex) * x_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_noise, noise.data(), sizeof(cuFloatComplex) * y_size, cudaMemcpyHostToDevice);

    // GPU benchmark
    dim3 grid(n_symbols);
    dim3 block(n_subcarriers);  // 8192 threads per block OK with newer cards
    cudaDeviceSynchronize();
    t1 = high_resolution_clock::now();
    compute_y_kernel<<<grid, block>>>(d_H, d_x, d_noise, d_y, n_symbols, n_subcarriers, n_rx, n_tx);
    cudaDeviceSynchronize();
    t2 = high_resolution_clock::now();
    float gpu_time = duration<float>(t2 - t1).count();

    // Copy result back (optional)
    cudaMemcpy(y_gpu.data(), d_y, sizeof(cuFloatComplex) * y_size, cudaMemcpyDeviceToHost);



    // Print benchmark
    cout << "Size: " << n_subcarriers
         << " | CPU: " << cpu_time << " s"
         << " | GPU: " << gpu_time << " s"
         << " | Speedup: " << (cpu_time / gpu_time) << "x" << endl;

    cudaFree(d_H); cudaFree(d_x); cudaFree(d_noise); cudaFree(d_y);
    return 0;
}

