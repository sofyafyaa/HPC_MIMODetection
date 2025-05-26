import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.modulations import qam_modulate, qam_demodulate
from src.detection import zf_detector, mmse_detector
from src.signals import generate_bits, generate_channel, generate_noise


class MIMOOFDMSimulation:
    def __init__(self, config):
        self.n_tx = config['n_tx']
        self.n_rx = config['n_rx']
        self.n_subcarriers = config['n_subcarriers']
        self.mod_order = config['mod_order']
        self.snr_db_range = config['snr_db_range']
        self.n_symbols = config['n_symbols']
        self.bits_per_symbol = int(np.log2(self.mod_order))

    def simulate(self):
        ber_zf = []
        ber_mmse = []

        for snr_db in tqdm(self.snr_db_range, leave=False):

            bits = generate_bits(self.n_symbols, self.n_tx, self.n_subcarriers, self.bits_per_symbol)
            symbols = qam_modulate(bits, self.mod_order)  # [n_symbols, n_subcarriers, n_tx]

            tx_signal = np.fft.ifft(symbols, axis=1).astype(np.complex64)

            snr_linear = 10 ** (snr_db / 10)
            noise_var = 1 / snr_linear

            H = generate_channel(self.n_symbols, self.n_rx, self.n_tx, self.n_subcarriers)
            noise = generate_noise(self.n_symbols, self.n_rx, self.n_subcarriers, noise_var)

            # y = H @ x
            y_time = np.matmul(H, tx_signal[..., np.newaxis])[..., 0] + noise

            y_freq = np.fft.fft(y_time, axis=1).astype(np.complex64)

            x_hat_zf = zf_detector(H, y_time)
            x_hat_mmse = mmse_detector(H, y_time, noise_var)

            x_hat_freq_zf = np.fft.fft(x_hat_zf, axis=1).astype(np.complex64)
            x_hat_freq_mmse = np.fft.fft(x_hat_mmse, axis=1).astype(np.complex64)

            bits_hat_zf = qam_demodulate(x_hat_freq_zf, self.mod_order)
            bits_hat_mmse = qam_demodulate(x_hat_freq_mmse, self.mod_order)

            total_bits = np.prod(bits.shape)
            bit_errors_zf = np.sum(bits != bits_hat_zf)
            bit_errors_mmse = np.sum(bits != bits_hat_mmse)

            ber_zf.append(bit_errors_zf / total_bits)
            ber_mmse.append(bit_errors_mmse / total_bits)

        self.plot_ber(ber_zf, ber_mmse)
        self.plot_constellations(symbols, x_hat_zf, x_hat_mmse)

    def plot_ber(self, ber_zf, ber_mmse):
        plt.figure()
        plt.semilogy(self.snr_db_range, ber_zf, '--', label='ZF')
        plt.semilogy(self.snr_db_range, ber_mmse, '--', label='MMSE')
        plt.xlabel('SNR (dB)')
        plt.ylabel('BER')
        plt.title(f'BER(SNR)')
        plt.grid(True, which='both')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_constellations(self, tx_symbols, rx_zf, rx_mmse):
        plt.figure(figsize=(15, 4))

        plt.subplot(1, 3, 1)
        plt.plot(tx_symbols[0, :, 0].real, tx_symbols[0, :, 0].imag, 'bo', label='TX')
        plt.title('Transmitted Symbols (TX)')
        plt.grid(True)
        plt.axis('equal')

        plt.subplot(1, 3, 2)
        plt.plot(rx_zf[0, :, 0].real, rx_zf[0, :, 0].imag, 'rx', label='RX ZF')
        plt.title('Received Symbols ZF')
        plt.grid(True)
        plt.axis('equal')

        plt.subplot(1, 3, 3)
        plt.plot(rx_mmse[0, :, 0].real, rx_mmse[0, :, 0].imag, 'gx', label='RX MMSE')
        plt.title('Received Symbols MMSE')
        plt.grid(True)
        plt.axis('equal')

        plt.tight_layout()
        plt.show()
