'''
The decoder module, responsible for decoding the carrier audio signal to text.
Liang Sidi, 2024
'''
import wave
import gzip
import numpy as np

class Demodulator:
    """ Class for demodulating carrier audio signals to text.
    Attributes:
        carrier_freq (int): The frequency of the carrier signal in Hz.
        sample_rate (int): The sampling rate in Hz.
        bit_duration (float): The duration of each bit in seconds.

    Methods:
        __init__(self, 
                carrier_freq=2000, 
                sample_rate=44100, 
                bit_duration=0.1): Initializes the demodulator.
        read_from_wav(self, filename): Reads a signal from a WAV file.
        bpsk_demodulate(self, signal): Demodulates the BPSK signal to binary data.
        binary_to_string(self, binary_data): Converts binary data to text.
        demodulate(self, signal, mode=1): Demodulates the signal to text.
    """
    def __init__(self, carrier_freq=2200, sample_rate=44100, bit_duration=0.1, bandwidth=2200):
        """Initialize the demodulator with specified parameters.
        Parameters:
            carrier_freq (int, default=2000): The frequency of the carrier signal in Hz.
            sample_rate (int, default=44100): The sampling rate in Hz.
            bit_duration (float, default=0.1): The duration of each bit in seconds.
            bandwidth (int, default=2200): The bandwidth of the signal in Hz.
        """
        self.carrier_freq = carrier_freq
        self.sample_rate = sample_rate
        self.bit_duration = bit_duration
        self.bandwidth = bandwidth

    def read_from_wav(self, filename):
        """Read a signal from a WAV file.
        Parameters:
            filename (str): The filename of the WAV file.
        Returns:
            The signal as a NumPy array.
        """
        with wave.open(filename, 'r') as wav_file:
            # Ensure the parameters match those used in modulation
            assert wav_file.getframerate() == self.sample_rate
            assert wav_file.getnchannels() == 1  # Mono
            assert wav_file.getsampwidth() == 2  # 16-bit samples

            # Read the frame data and convert it to a NumPy array
            frames = wav_file.readframes(wav_file.getnframes())
            signal = np.frombuffer(frames, dtype=np.int16)

        # Normalize the signal
        return signal / max(abs(signal))

    def mfsk_demodulate(self, signal, M=64):
        """Demodulate the MFSK signal to binary data.
        Parameters: 
            signal (np.array): The signal to demodulate.
            M (int): The M in MFSK, representing the number of different frequencies. 
                     Must be a power of 2.
        Returns:
            The binary data.
        """
        bits_per_symbol = int(np.log2(M))
        num_symbols = len(signal) // int(self.sample_rate * self.bit_duration)

        # Frequencies for each symbol
        # symbol_freqs = np.linspace(self.carrier_freq - M*440/2, self.carrier_freq + M*440/2, M)
        symbol_freqs = np.linspace(self.carrier_freq - self.bandwidth/2,
                                   self.carrier_freq + self.bandwidth/2,
                                   M)
        # print(symbol_freqs)

        binary_data = ""
        for i in range(num_symbols):
            start = i * int(self.sample_rate * self.bit_duration)
            end = start + int(self.sample_rate * self.bit_duration)
            symbol_slice = signal[start:end]

            # Zero padding for increased FFT resolution
            zero_padded_slice = np.pad(symbol_slice, (0, len(symbol_slice)), 'constant')

            # Perform FFT
            fft_result = np.fft.fft(zero_padded_slice)
            freqs = np.fft.fftfreq(len(zero_padded_slice), 1 / self.sample_rate)
            
            # Focus on the positive frequencies only
            half_n = len(fft_result) // 2
            fft_result_positive = fft_result[:half_n]
            freqs_positive = freqs[:half_n]

            # Find the frequency in symbol_freqs closest to the peak frequency in the FFT
            peak_freq = freqs_positive[np.argmax(np.abs(fft_result_positive))]
            # print(peak_freq)
            symbol = np.argmin(np.abs(symbol_freqs - peak_freq))
            binary_data += format(symbol, f'0{int(np.log2(M))}b')

        return binary_data
    
    def bpsk_demodulate(self, signal):
        """Demodulate the BPSK signal to binary data.
        Parameters: 
            signal (np.array): The signal to demodulate.
        Returns:
            The binary data.
        """
        num_samples_per_bit = int(self.sample_rate * self.bit_duration)
        num_bits = len(signal) // num_samples_per_bit
        binary_data = ""

        for i in range(num_bits):
            start = i * num_samples_per_bit
            end = start + num_samples_per_bit
            bit_slice = signal[start:end]
            t = np.linspace(start / self.sample_rate,
                            end / self.sample_rate,
                            end - start,
                            endpoint=False)
            reference_signal = np.cos(2 * np.pi * 2 * self.carrier_freq * t)
            product = np.mean(bit_slice * reference_signal)
            binary_data += '1' if product > 0 else '0'

        return binary_data

    def binary_to_string(self, binary_data, gzip_enabled=True):
        """Convert binary data to text.
        Parameters:
            binary_data(str): The binary data to be converted.
            gzip_enabled(bool, default=True): Whether to use gzip compression.
        Returns:
            The text.
        """
        byte_array = bytearray(int(binary_data[i:i+8], 2) for i in range(0, len(binary_data), 8))
        if gzip_enabled:
            byte_array = gzip.decompress(byte_array)
        return byte_array.decode('utf-8', errors='ignore')

    def demodulate(self, signal, mode=2):
        """Demodulate the signal to text.
        Parameters:
            signal(np.array): The signal to demodulate.
            mode(int, default=2): The demodulation mode. 
                                  Currently only supports BPSK (1), 64-FSK (2).
        Returns:
            The text.
        """
        if mode == 1:
            binary_data = self.bpsk_demodulate(signal)
        elif mode == 2:
            binary_data = self.mfsk_demodulate(signal) # 64-FSK
        else:
            # Not Implemented
            return None
        return self.binary_to_string(binary_data)


# Example usage
demodulator = Demodulator()
signal_64fsk = demodulator.read_from_wav("hello_world_64fsk.wav")
decoded_text_64fsk = demodulator.demodulate(signal_64fsk)
print("64-FSK: \n", decoded_text_64fsk)

signal_64fsk_gaussian = demodulator.read_from_wav("hello_world_64fsk_gaussian.wav")
decoded_text_64fsk_gaussian = demodulator.demodulate(signal_64fsk_gaussian)
print("64-FSK with Gaussian Noise: \n", decoded_text_64fsk_gaussian)

signal_bpsk = demodulator.read_from_wav("hello_world_bpsk.wav")
decoded_text_bpsk = demodulator.demodulate(signal_bpsk, mode=1)
print("BPSK: \n", decoded_text_bpsk)

signal_bpsk_gaussian = demodulator.read_from_wav("hello_world_bpsk_gaussian.wav")
decoded_text_bpsk_gaussian = demodulator.demodulate(signal_bpsk_gaussian, mode=1)
print("BPSK with Gaussian Noise: \n", decoded_text_bpsk_gaussian)

