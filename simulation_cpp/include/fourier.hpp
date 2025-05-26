#ifndef FOURIER_HPP
#define FOURIER_HPP

#include <vector>
#include <complex>

using Complex = std::complex<float>;
using ComplexVector = std::vector<Complex>;

void apply_fft(ComplexVector& data);

void apply_ifft(ComplexVector& data);

#endif
