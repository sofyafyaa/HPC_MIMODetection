#include <iostream>
#include <vector>
#include <complex>
#include <random>
#include <cmath>
#include <chrono>
#include <fstream>
#include <fftw3.h>
#include <omp.h>


using namespace std;
using Complex = complex<float>;
using namespace chrono;


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


void generate_bits(vector<vector<vector<int>>> &bits, int n_symbols, int n_tx, int n_subcarriers, int k) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, 1);
    for (int s = 0; s < n_symbols; ++s)
        for (int tx = 0; tx < n_tx; ++tx)
            for (int i = 0; i < n_subcarriers * k; ++i)
                bits[s][tx][i] = dis(gen);
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


vector<vector<vector<Complex>>> zf_detector(
    const vector<vector<vector<vector<Complex>>>> &H,
    const vector<vector<vector<Complex>>> &y) {

    int n_symbols = y.size();
    int n_subc = y[0].size();
    int n_rx = y[0][0].size();
    int n_tx = H[0][0][0].size();

    vector<vector<vector<Complex>>> x_hat(n_symbols, vector<vector<Complex>>(n_subc, vector<Complex>(n_tx)));

    for (int s = 0; s < n_symbols; ++s) {
        for (int sc = 0; sc < n_subc; ++sc) {
            // Form matrix H and pseudo-inverse
            vector<vector<Complex>> h(n_rx, vector<Complex>(n_tx));
            for (int rx = 0; rx < n_rx; ++rx)
                for (int tx = 0; tx < n_tx; ++tx)
                    h[rx][tx] = H[s][sc][rx][tx];

            // Use pseudo-inverse (not optimized, replace with LAPACK later)
            for (int tx = 0; tx < n_tx; ++tx) {
                Complex sum = 0.0;
                for (int rx = 0; rx < n_rx; ++rx)
                    sum += conj(h[rx][tx]) * y[s][sc][rx];
                x_hat[s][sc][tx] = sum; // crude ZF (non-normalized)
            }
        }
    }
    return x_hat;
}


float calculate_ber(const vector<vector<vector<int>>> &bits,
                    const vector<vector<vector<Complex>>> &rx_symbols,
                    int M) {
    int k = bits_per_symbol(M);
    vector<Complex> constell = generate_qam_constellation(M);
    int errors = 0, total = 0;

    for (size_t s = 0; s < bits.size(); ++s)
        for (size_t tx = 0; tx < bits[0].size(); ++tx)
            for (size_t sc = 0; sc < bits[0][0].size() / k; ++sc) {
                Complex sym = rx_symbols[s][sc][tx];
                int min_i = 0;
                float min_dist = abs(sym - constell[0]);
                for (int i = 1; i < M; ++i) {
                    float d = abs(sym - constell[i]);
                    if (d < min_dist) {
                        min_dist = d;
                        min_i = i;
                    }
                }
                vector<int> demod_bits = decimal_to_bits(min_i, k);
                for (int b = 0; b < k; ++b) {
                    int orig_bit = bits[s][tx][sc * k + b];
                    if (demod_bits[b] != orig_bit) ++errors;
                    ++total;
                }
            }
    return static_cast<float>(errors) / total;
}



int main() {
    const int n_tx = 2, n_rx = 2;
    const int n_subcarriers = 64;
    const int mod_order = 16;
    const int n_symbols = 100;
    const vector<int> snr_db_range = {0, 5, 10, 15, 20, 25, 30};
    const int k = bits_per_symbol(mod_order);

    ofstream fout("../results/ber_results.csv");
    fout << "SNR_dB,BER_ZF\n";

    for (int snr_db : snr_db_range) {
        float snr_lin = pow(10.0, snr_db / 10.0);
        float noise_var = 1.0 / snr_lin;


        //////////////// TX ///////////////
        vector<vector<vector<int>>> bits(n_symbols, vector<vector<int>>(n_tx, vector<int>(n_subcarriers * k)));
        generate_bits(bits, n_symbols, n_tx, n_subcarriers, k);
        auto symbols = qam_modulate(bits, mod_order);
        //////////////// TX ///////////////


        ///////////// CHANNEL /////////////
        auto H = generate_channel(n_symbols, n_rx, n_tx, n_subcarriers);
        auto noise = generate_noise(n_symbols, n_rx, n_subcarriers, noise_var);
        ///////////// CHANNEL /////////////


        //////////////// RX ///////////////
        vector<vector<vector<Complex>>> y(n_symbols, vector<vector<Complex>>(n_subcarriers, vector<Complex>(n_rx)));
        for (int s = 0; s < n_symbols; ++s)
            for (int sc = 0; sc < n_subcarriers; ++sc)
                for (int rx = 0; rx < n_rx; ++rx) {
                    y[s][sc][rx] = noise[s][sc][rx];
                    for (int tx = 0; tx < n_tx; ++tx)
                        y[s][sc][rx] += H[s][sc][rx][tx] * symbols[s][sc][tx];
                }
        //////////////// RX ///////////////


        //////////// DETECTION ////////////
        auto start = high_resolution_clock::now();
        auto x_hat_zf = zf_detector(H, y);
        auto end = high_resolution_clock::now();
        float time_zf = duration<float>(end - start).count();
        //////////// DETECTION ////////////


        float ber_zf = calculate_ber(bits, x_hat_zf, mod_order);
        fout << snr_db << "," << ber_zf << endl;
        cout << "SNR: " << snr_db << " dB, ZF-BER: " << ber_zf << ", Time: " << time_zf << " sec" << endl;
    }

    fout.close();
    return 0;
}
