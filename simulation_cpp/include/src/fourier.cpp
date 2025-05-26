#include "Fourier.hpp"
#include <fftw3.h>

void apply_fft(ComplexVector& data) {
    int N = data.size();
    fftwf_complex* in = reinterpret_cast<fftwf_complex*>(data.data());
    fftwf_complex* out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N);

    fftwf_plan p = fftwf_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(p);

    for (int i = 0; i < N; ++i)
        data[i] = Complex(out[i][0], out[i][1]);

    fftwf_destroy_plan(p);
    fftwf_free(out);
}

void apply_ifft(ComplexVector& data) {
    int N = data.size();
    fftwf_complex* in = reinterpret_cast<fftwf_complex*>(data.data());
    fftwf_complex* out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N);

    fftwf_plan p = fftwf_plan_dft_1d(N, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftwf_execute(p);

    for (int i = 0; i < N; ++i)
        data[i] = Complex(out[i][0], out[i][1]) / static_cast<float>(N);  // normalization

    fftwf_destroy_plan(p);
    fftwf_free(out);
}
