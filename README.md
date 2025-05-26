# MIMO-OFDM signal detection via OpenMP and CUDA

Our project implement MIMO-OFDM signal detection using two algorithms: Zero-Forcing and Minimum Mean Square Error. 
The implementation includes high-performance versions using OpenMP and CUDA.


### Zero-Forcing (ZF)

By computing the pseudo-inverse of the channel matrix \( H \):

$$
\hat{x}_{\text{ZF}} = \left( H^H H \right)^{-1} H^H y = H^{\dagger} y
$$

### Minimum Mean Square Error (MMSE)

$$
\hat{x}_{\text{MMSE}} = \left( H^H H + \sigma^2 I \right)^{-1} H^H y
$$

---

## Simulation Pipeline

1) Bit signal generation
2) QAM symbol modulation
3) IFFT
4) Signal transmitions
5) Channel H
6) Amplifiers AWGN noise (SNR)
7) Signal receiving
8) FFT
9) ZF/MMSE detection
10) QAM symbol demodulation
11) BER calculation
