#ifndef DETECTION_HPP
#define DETECTION_HPP

#include <vector>
#include <complex>

using namespace std;
using Complex = complex<float>;

vector<vector<vector<Complex>>> zf_detector(
    const vector<vector<vector<vector<Complex>>>> &H,
    const vector<vector<vector<Complex>>> &y);

vector<vector<vector<Complex>>> mmse_detector(
    const vector<vector<vector<vector<Complex>>>> &H,
    const vector<vector<vector<Complex>>> &y,
    float noise_var);

#endif // DETECTION_HPP
