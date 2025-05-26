#ifndef GENERATION_HPP
#define GENERATION_HPP

#include <vector>
#include <complex>

using namespace std;
using Complex = complex<float>;

void generate_bits(vector<vector<vector<int>>> &bits, int n_symbols, int n_tx, int n_subcarriers, int k);
vector<vector<vector<vector<Complex>>>> generate_channel(int n_symbols, int n_rx, int n_tx, int n_subcarriers);
vector<vector<vector<Complex>>> generate_noise(int n_symbols, int n_rx, int n_subcarriers, float noise_var);

#endif // GENERATION_HPP
