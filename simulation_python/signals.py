import numpy as np


def generate_bits(n_symbols, n_tx, n_subcarriers, bits_per_symbol):
    '''
    random generation of transmitted signal
    n_tx -- number of TX antennas
    n_subcarriers -- number of subcarries in channel
    bits_per_symbol -- depth of QAM constellation
    '''
    return np.random.randint(0, 2, size=(n_symbols, n_tx, n_subcarriers * bits_per_symbol))


def generate_channel(n_symbols, n_rx, n_tx, n_subcarriers):
    '''
    generation of synthetic channel matrix H
    n_rx -- number of rx antennas
    n_tx -- number of tx antennas
    n_subcarriers -- number of subcarries in channel
    '''
    return (np.random.randn(n_symbols, n_subcarriers, n_rx, n_tx) +
            1j * np.random.randn(n_symbols, n_subcarriers, n_rx, n_tx)) / np.sqrt(2)


def generate_noise(n_symbols, n_rx, n_subcarriers, noise_var):
    '''
    generation of AWGN noise in channel
    n_rx -- number of rx antennas
    n_subcarriers -- number of subcarries in channel
    noise_var -- noise distribution sigma
    '''
    return (np.random.randn(n_symbols, n_subcarriers, n_rx) +
            1j * np.random.randn(n_symbols, n_subcarriers, n_rx)) * np.sqrt(noise_var / 2)