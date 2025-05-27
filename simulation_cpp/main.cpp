#include <iostream>
#include <vector>
#include <complex>
#include <random>
#include <cmath>
#include <chrono>
#include <fstream>
#include <fftw3.h>
#include <omp.h>
#include "generation.hpp"
#include "modulation.hpp"
#include "detection.hpp"
#include "fourier.hpp"

using namespace std;
using Complex = complex<float>;
using namespace chrono;


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
    /////////////// CONFIG ///////////////
    const int n_tx = 4;
    const int n_rx = 64;
    const int n_subcarriers = 128;
    const int mod_order = 4; // [2, 4, 16]
    const int n_symbols = 200;
    const int k = bits_per_symbol(mod_order);

    int start_snr = -10;
    int end_snr = 25;
    int step_snr = 5;
    vector<int> snr_db_range;
    for (int snr = start_snr; snr <= end_snr; snr += step_snr) {
        snr_db_range.push_back(snr);
    }

    ofstream fout("../results/ber_results.csv");
    fout << "SNR_dB,BER_ZF,BER_MMSE,Time_IFFT,Time_Channel,Time_ZF,Time_MMSE,Time_FFT\n";
    /////////////// CONFIG ///////////////


    ///////////////// SIMULATION /////////////////
    for (int snr_db : snr_db_range) {
        float snr_lin = pow(10.0, snr_db / 10.0);
        float noise_var = 1.0 / snr_lin;

        //////////////// TX ///////////////
        // generate bit signal
        vector<vector<vector<int>>> bits(n_symbols, vector<vector<int>>(n_tx, vector<int>(n_subcarriers * k)));
        generate_bits(bits, n_symbols, n_tx, n_subcarriers, k);
        // 16-QAM modulation
        auto symbols = qam_modulate(bits, mod_order);

        auto t1 = high_resolution_clock::now();
        // IFFT
        for (int s = 0; s < n_symbols; ++s)
            for (int tx = 0; tx < n_tx; ++tx) {
                vector<Complex> time_domain(n_subcarriers);
                for (int sc = 0; sc < n_subcarriers; ++sc)
                    time_domain[sc] = symbols[s][sc][tx];

                apply_ifft(time_domain);

                for (int sc = 0; sc < n_subcarriers; ++sc)
                    symbols[s][sc][tx] = time_domain[sc];
            }
        auto t2 = high_resolution_clock::now();
        float time_ifft = duration<float>(t2 - t1).count();
        //////////////// TX ///////////////


        ///////////// CHANNEL /////////////
        // channel matrix H [tx, rx]
        auto H = generate_channel(n_symbols, n_rx, n_tx, n_subcarriers);
        // AWGN noise
        auto noise = generate_noise(n_symbols, n_rx, n_subcarriers, noise_var);
        ///////////// CHANNEL /////////////


        //////////////// RX ///////////////
        t1 = high_resolution_clock::now();
        // y = Hx + n
        vector<vector<vector<Complex>>> y(n_symbols, vector<vector<Complex>>(n_subcarriers, vector<Complex>(n_rx)));
        for (int s = 0; s < n_symbols; ++s)
            for (int sc = 0; sc < n_subcarriers; ++sc)
                for (int rx = 0; rx < n_rx; ++rx) {
                    y[s][sc][rx] = noise[s][sc][rx];
                    for (int tx = 0; tx < n_tx; ++tx)
                        y[s][sc][rx] += H[s][sc][rx][tx] * symbols[s][sc][tx];
                }
        t2 = high_resolution_clock::now();
        float time_channel = duration<float>(t2 - t1).count();
        //////////////// RX ///////////////


        //////////// DETECTION ////////////
        t1 = high_resolution_clock::now();
        // x_est = H^(-1)y
        auto x_hat_zf = zf_detector(H, y);
        t2 = high_resolution_clock::now();
        float time_zf = duration<float>(t2 - t1).count();

        t1 = high_resolution_clock::now();
        // x_est = inv(H^H@H + sigma^2*I) @ H^H y
        auto x_hat_mmse = mmse_detector(H, y, noise_var);
        t2 = high_resolution_clock::now();
        float time_mmse = duration<float>(t2 - t1).count();

        t1 = high_resolution_clock::now();
        // FFT
        for (int s = 0; s < n_symbols; ++s)
            for (int tx = 0; tx < n_tx; ++tx) {
                vector<Complex> time_domain(n_subcarriers);
                for (int sc = 0; sc < n_subcarriers; ++sc)
                    time_domain[sc] = x_hat_zf[s][sc][tx];

                apply_fft(time_domain);

                for (int sc = 0; sc < n_subcarriers; ++sc)
                    x_hat_zf[s][sc][tx] = time_domain[sc];
            }
        t2 = high_resolution_clock::now();
        float time_fft = duration<float>(t2 - t1).count();


        for (int s = 0; s < n_symbols; ++s)
            for (int tx = 0; tx < n_tx; ++tx) {
                vector<Complex> time_domain(n_subcarriers);
                for (int sc = 0; sc < n_subcarriers; ++sc)
                    time_domain[sc] = x_hat_mmse[s][sc][tx];

                apply_fft(time_domain);

                for (int sc = 0; sc < n_subcarriers; ++sc)
                    x_hat_mmse[s][sc][tx] = time_domain[sc];
            }

        //////////// DETECTION ////////////

        float ber_zf = calculate_ber(bits, x_hat_zf, mod_order);
        float ber_mmse = calculate_ber(bits, x_hat_mmse, mod_order);

        fout << snr_db << "," << ber_zf << "," << ber_mmse << "," << time_ifft << "," 
             << time_channel << "," << time_zf << "," << time_mmse << "," << time_fft << "\n";

        cout << "SNR: " << snr_db 
             << " | BER_ZF: " << ber_zf
             << " | BER_MMSE: " << ber_mmse
             << " | IFFT: " << time_ifft << "s"
             << " | Channel: " << time_channel << "s"
             << " | ZF: " << time_zf << "s"
             << " | MMSE: " << time_mmse << "s"
             << " | FFT: " << time_fft << "s" << endl;
    }
    ///////////////// SIMULATION /////////////////

    fout.close();
    return 0;
}
