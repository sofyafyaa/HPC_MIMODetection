#include "detection.hpp"
#include "modulation.hpp"

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
            vector<vector<Complex>> h(n_rx, vector<Complex>(n_tx));
            for (int rx = 0; rx < n_rx; ++rx)
                for (int tx = 0; tx < n_tx; ++tx)
                    h[rx][tx] = H[s][sc][rx][tx];

            for (int tx = 0; tx < n_tx; ++tx) {
                Complex sum = 0.0;
                for (int rx = 0; rx < n_rx; ++rx)
                    sum += conj(h[rx][tx]) * y[s][sc][rx];
                x_hat[s][sc][tx] = sum;
            }
        }
    }
    return x_hat;
}

vector<vector<vector<Complex>>> mmse_detector(
    const vector<vector<vector<vector<Complex>>>> &H,
    const vector<vector<vector<Complex>>> &y,
    float noise_var) {

    int n_symbols = y.size();
    int n_subc = y[0].size();
    int n_rx = y[0][0].size();
    int n_tx = H[0][0][0].size();

    vector<vector<vector<Complex>>> x_hat(n_symbols, vector<vector<Complex>>(n_subc, vector<Complex>(n_tx)));

    for (int s = 0; s < n_symbols; ++s) {
        for (int sc = 0; sc < n_subc; ++sc) {
            vector<vector<Complex>> Hmat(n_rx, vector<Complex>(n_tx));
            for (int rx = 0; rx < n_rx; ++rx)
                for (int tx = 0; tx < n_tx; ++tx)
                    Hmat[rx][tx] = H[s][sc][rx][tx];

            vector<vector<Complex>> HH(n_tx, vector<Complex>(n_tx, 0));
            for (int i = 0; i < n_tx; ++i)
                for (int j = 0; j < n_tx; ++j)
                    for (int rx = 0; rx < n_rx; ++rx)
                        HH[i][j] += conj(Hmat[rx][i]) * Hmat[rx][j];

            for (int i = 0; i < n_tx; ++i)
                HH[i][i] += noise_var;

            vector<vector<Complex>> inv(HH);
            vector<vector<Complex>> I(n_tx, vector<Complex>(n_tx, 0));
            for (int i = 0; i < n_tx; ++i)
                I[i][i] = 1;

            for (int i = 0; i < n_tx; ++i) {
                Complex pivot = inv[i][i];
                for (int j = 0; j < n_tx; ++j) {
                    inv[i][j] /= pivot;
                    I[i][j] /= pivot;
                }
                for (int k = 0; k < n_tx; ++k) {
                    if (k != i) {
                        Complex factor = inv[k][i];
                        for (int j = 0; j < n_tx; ++j) {
                            inv[k][j] -= factor * inv[i][j];
                            I[k][j] -= factor * I[i][j];
                        }
                    }
                }
            }

            vector<Complex> Hy(n_tx, 0);
            for (int i = 0; i < n_tx; ++i)
                for (int rx = 0; rx < n_rx; ++rx)
                    Hy[i] += conj(Hmat[rx][i]) * y[s][sc][rx];

            for (int i = 0; i < n_tx; ++i) {
                x_hat[s][sc][i] = 0;
                for (int j = 0; j < n_tx; ++j)
                    x_hat[s][sc][i] += I[i][j] * Hy[j];
            }
        }
    }
    return x_hat;
}
