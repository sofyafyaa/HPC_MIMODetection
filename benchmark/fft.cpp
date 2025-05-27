#include <iostream>
#include <vector>
#include <complex>
#include <chrono>
#include <fftw3.h>
#include <cuda_runtime.h>
#include <cufft.h>

using namespace std;
using namespace chrono;
using Complex = complex<float>;

int main() {
    const int N = 1 << 16; // 65536

    // Allocate host memory
    vector<Complex> data(N);
    for (int i = 0; i < N; ++i)
        data[i] = {static_cast<float>(rand()) / RAND_MAX, 0.0f};

    ///////////////////////// FFTW /////////////////////////
    vector<Complex> fftw_data = data;
    fftwf_complex* in_fftw  = reinterpret_cast<fftwf_complex*>(fftw_data.data());
    fftwf_complex* out_fftw = reinterpret_cast<fftwf_complex*>(fftw_data.data());
    auto plan_fftw = fftwf_plan_dft_1d(N, in_fftw, out_fftw, FFTW_FORWARD, FFTW_ESTIMATE);

    auto t1 = high_resolution_clock::now();
    fftwf_execute(plan_fftw);
    auto t2 = high_resolution_clock::now();
    float fftw_time = duration<float>(t2 - t1).count();
    fftwf_destroy_plan(plan_fftw);

    ///////////////////////// cuFFT /////////////////////////
    cufftHandle plan_cufft;
    cufftComplex* d_data;

    cudaMalloc(&d_data, sizeof(cufftComplex) * N);
    cudaMemcpy(d_data, data.data(), sizeof(cufftComplex) * N, cudaMemcpyHostToDevice);
    cufftPlan1d(&plan_cufft, N, CUFFT_C2C, 1);

    cudaDeviceSynchronize(); // ensure accurate timing
    t1 = high_resolution_clock::now();
    cufftExecC2C(plan_cufft, d_data, d_data, CUFFT_FORWARD);
    cudaDeviceSynchronize();
    t2 = high_resolution_clock::now();
    float cufft_time = duration<float>(t2 - t1).count();

    cufftDestroy(plan_cufft);
    cudaFree(d_data);

    ///////////////////////// RESULT /////////////////////////
    cout << "FFTW Time : " << fftw_time << " s\n";
    cout << "cuFFT Time: " << cufft_time << " s\n";
    cout << "Speedup   : " << (fftw_time / cufft_time) << "x\n";

    return 0;
}
