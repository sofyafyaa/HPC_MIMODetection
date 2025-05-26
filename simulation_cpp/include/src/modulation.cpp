#include "modulation.hpp"
#include <cmath>
#include <random>

int bits_per_symbol(int M) {
    return static_cast<int>(log2(M));
}

vector<Complex> generate_qam_constellation(int M) {
    int m = static_cast<int>(sqrt(M));
    vector<Complex> constellation;
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
            constellation.emplace_back(2 * j - m + 1, 2 * i - m + 1);

    float norm_val = 0.0;
    for (auto &c : constellation)
        norm_val += norm(c);
    norm_val = sqrt(norm_val / M);

    for (auto &c : constellation)
        c /= norm_val;

    return constellation;
}

vector<int> bits_to_decimal(const vector<int> &bits, int k) {
    int n = bits.size() / k;
    vector<int> dec(n);
    for (int i = 0; i < n; ++i) {
        dec[i] = 0;
        for (int j = 0; j < k; ++j)
            dec[i] = (dec[i] << 1) | bits[i * k + j];
    }
    return dec;
}

vector<int> decimal_to_bits(int val, int k) {
    vector<int> bits(k, 0);
    for (int i = k - 1; i >= 0; --i) {
        bits[i] = val & 1;
        val >>= 1;
    }
    return bits;
}

vector<vector<vector<Complex>>> qam_modulate(const vector<vector<vector<int>>> &bits, int M) {
    int n_symbols = bits.size();
    int n_tx = bits[0].size();
    int total_bits = bits[0][0].size();
    int k = bits_per_symbol(M);
    int n_subcarriers = total_bits / k;
    vector<Complex> constellation = generate_qam_constellation(M);
    vector<vector<vector<Complex>>> symbols(n_symbols, vector<vector<Complex>>(n_subcarriers, vector<Complex>(n_tx)));

    for (int s = 0; s < n_symbols; ++s) {
        for (int tx = 0; tx < n_tx; ++tx) {
            for (int sc = 0; sc < n_subcarriers; ++sc) {
                int val = 0;
                for (int b = 0; b < k; ++b)
                    val = (val << 1) | bits[s][tx][sc * k + b];
                symbols[s][sc][tx] = constellation[val];
            }
        }
    }
    return symbols;
}
