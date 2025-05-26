#include "generation.hpp"
#include <random>

void generate_bits(vector<vector<vector<int>>> &bits, int n_symbols, int n_tx, int n_subcarriers, int k) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, 1);
    for (int s = 0; s < n_symbols; ++s)
        for (int tx = 0; tx < n_tx; ++tx)
            for (int i = 0; i < n_subcarriers * k; ++i)
                bits[s][tx][i] = dis(gen);
}

vector<vector<vector<vector<Complex>>>> generate_channel(int n_symbols, int n_rx, int n_tx, int n_subcarriers) {
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<float> dist(0.0, 1.0);

    vector<vector<vector<vector<Complex>>>> H(n_symbols, vector<vector<vector<Complex>>>(
        n_subcarriers, vector<vector<Complex>>(n_rx, vector<Complex>(n_tx))));

    for (int s = 0; s < n_symbols; ++s)
        for (int sc = 0; sc < n_subcarriers; ++sc)
            for (int rx = 0; rx < n_rx; ++rx)
                for (int tx = 0; tx < n_tx; ++tx)
                    H[s][sc][rx][tx] = Complex(dist(gen), dist(gen)) / static_cast<float>(sqrt(2.0));
    return H;
}

vector<vector<vector<Complex>>> generate_noise(int n_symbols, int n_rx, int n_subcarriers, float noise_var) {
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<float> dist(0.0, sqrt(noise_var / 2));

    vector<vector<vector<Complex>>> noise(n_symbols, vector<vector<Complex>>(n_subcarriers, vector<Complex>(n_rx)));
    for (int s = 0; s < n_symbols; ++s)
        for (int sc = 0; sc < n_subcarriers; ++sc)
            for (int rx = 0; rx < n_rx; ++rx)
                noise[s][sc][rx] = Complex(dist(gen), dist(gen));
    return noise;
}
