import numpy as np


def qam_constellation(M):
    '''
    Generate M-QAM modulation for signal
    Parameters:
    - M: modulation order
    '''
    m = int(np.sqrt(M))
    re = 2 * (np.arange(m) - m / 2 + 0.5)
    im = 2 * (np.arange(m) - m / 2 + 0.5)
    const = np.array([x + 1j*y for y in im for x in re])
    norm_factor = np.sqrt((np.abs(const) ** 2).mean())
    return const / norm_factor


def qam_modulate(bits, M):
    """
    Encode bit signal into QAM symbols

    Parameters:
    - M: modulation order
    - bits: binary signal
    
    Returns:
    - symbols: [n_symbols, n_subcarriers, n_tx]
    """
    k = int(np.log2(M))
    n_symbols, n_tx, total_bits = bits.shape
    n_subcarriers = total_bits // k

    bits_reshaped = bits.reshape(n_symbols, n_tx, n_subcarriers, k)
    bits_decimal = bits_reshaped.dot(1 << np.arange(k)[::-1])
    const = qam_constellation(M)
    symbols = const[bits_decimal]
    return np.transpose(symbols, (0, 2, 1))


def qam_demodulate(symbols, M):
    """
    Hard-decision demodulation
    Parameters:
    - symbols: [n_symbols, n_subcarriers, n_tx]
    Returns:
    - bits: [n_symbols, n_tx, n_subcarriers * bits_per_symbol]
    """
    k = int(np.log2(M))
    const = qam_constellation(M)
    n_symbols, n_subcarriers, n_tx = symbols.shape
    symbols_flat = symbols.reshape(-1)
    dists = np.abs(symbols_flat[:, None] - const[None, :])
    nearest = dists.argmin(axis=1)
    bits = ((nearest[:, None] & (1 << np.arange(k)[::-1])) > 0).astype(int)
    bits = bits.reshape(n_symbols, n_subcarriers, n_tx, k)
    bits = np.transpose(bits, (0, 2, 1, 3))
    return bits.reshape(n_symbols, n_tx, n_subcarriers * k)