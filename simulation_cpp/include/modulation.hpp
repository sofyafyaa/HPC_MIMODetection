#ifndef MODULATION_HPP
#define MODULATION_HPP

#include <vector>
#include <complex>

using namespace std;
using Complex = complex<float>;

int bits_per_symbol(int M);
vector<Complex> generate_qam_constellation(int M);
vector<int> bits_to_decimal(const vector<int> &bits, int k);
vector<int> decimal_to_bits(int val, int k);
vector<vector<vector<Complex>>> qam_modulate(const vector<vector<vector<int>>> &bits, int M);

#endif // MODULATION_HPP
