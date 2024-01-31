'''
The decoder module, responsible for decoding the carrier audio signal to text.
Liang Sidi, 2024
'''
import wave
import gzip
import numpy as np
import scipy.signal

class Demodulator:
    """ Class for demodulating carrier audio signals to text.
    Attributes:
        carrier_freq (int): The frequency of the carrier signal in Hz.
        sample_rate (int): The sampling rate in Hz.
        bit_duration (float): The duration of each bit in seconds.
        bandwidth (int): The bandwidth of the signal in Hz.
        modulation_mode (int): The modulation mode. 1 for BPSK, 2
                               for MFSK.
        m_for_mfsk (int): The M in MFSK, default to 64-FSK. Must be a power of 2.
        compression_enabled (bool): Whether to use gzip compression.
        
    Methods:
        __init__(self, 
                carrier_freq=2000, 
                sample_rate=44100, 
                bit_duration=0.01,
                bandwidth=2000,
                modulation_mode=2,
                m_for_mfsk=64,
                compression_enabled=True): Initializes the demodulator.
        read_from_wav(self, filename): Reads a signal from a WAV file.
        get_text(self, signal): Recovers text from a signal.
        __bpsk_demodulate(self, signal): Demodulates the BPSK signal to binary data.
        __mfsk_demodulate(self, signal, M=64): Demodulates the MFSK signal to binary data.
        __binary_to_string(self, binary_data): Converts binary data to text.
        demodulate(self, signal, mode=1): Demodulates the signal to text.
    """
    def __init__(self, carrier_freq=8800, sample_rate=44100,
                 bit_duration=0.01, bandwidth=4400, modulation_mode=2,
                 m_for_mfsk=64, compression_enabled=True):
        """Initialize the demodulator with specified parameters.
        Parameters:
            carrier_freq (int, default=8800): The frequency of the carrier signal in Hz.
            sample_rate (int, default=44100): The sampling rate in Hz.
            bit_duration (float, default=0.01): The duration of each bit in seconds.
            bandwidth (int, default=4400): The bandwidth of the signal in Hz.
            modulation_mode (int, default=2): The modulation mode.
                                              1 for BPSK, 2 for MFSK.
            m_for_mfsk (int, default=64): The M in MFSK, default to 64-FSK. 
                                          Must be a power of 2.
            compression_enabled (bool, default=True): Whether to use gzip compression.
        """
        self.carrier_freq = carrier_freq
        self.sample_rate = sample_rate
        self.bit_duration = bit_duration
        self.bandwidth = bandwidth
        self.modulation_mode = modulation_mode
        # Check is M is a power of 2
        assert m_for_mfsk & (m_for_mfsk - 1) == 0 and m_for_mfsk != 0, "M must be a power of 2"
        self.m_for_mfsk = m_for_mfsk
        self.compression_enabled = compression_enabled
        self.start_chrip_seq = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25,
                         587.33, 659.25, 698.46, 783.99, 880.00, 987.77, 1046.50]
        self.end_chrip_seq = self.start_chrip_seq[::-1]
        self.preamble_duration = 1
        self.postamble_duration = 1

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

    def get_text(self, signal):
        """Recover text from a signal.
        Protocol: "HMMSG##<text>##MSGEND"
        Parameters:
            signal (np.array): The signal to recover text from.
        Returns:
            The text.
        """
        binary_data = self.__demodulate(signal)
        test_data = "0100100001001101010011010101001101000111001000110010001101010100011010000110010100100000011100010111010101101001011000110110101100100000011000100111001001101111011101110110111000100000011001100110111101111000001000000110101001110101011011010111000001110011001000000110111101110110011001010111001000100000011101000110100001100101001000000110110001100001011110100111100100100000011001000110111101100111001011100010000000001010001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000111001011010010010101001111001011001110010110000111001111000111010000100111010011011101110000100111011111011110010001100111001011010111010000111111001011010111010011001111001101011010010101010111010001000110110010010111011111011110010001100111001101001011110100101111001101001110010001000111001111001101110001000111001001011101110000100111011111011110010001100111010001011111010110000111001011010111010111111111001011000100010010111111001011011110010100000111011111011110010001100000010100010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000001110010110101111100100101110011010011101101001011110011010011010100100011110010110111110100000001110111110111100100011001110011110100111100010111110011010010100101101101110010110000110101011001110100010010111100011111110111110111100100011001110100110010111101100001110010010111101100110011110011010001000100100001110010110110010100000011110111110111100100011001110010110111110100010111110010110010000100101011110100010110000100000111110100110011000101100111110001110000000100000100010001100100011010011010101001101000111010001010100111001000100" 
        print(binary_data)
        return self.__binary_to_string(binary_data, gzip_enabled=self.compression_enabled)

    def __mfsk_demodulate(self, signal, M=64):
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

    def __bpsk_demodulate(self, signal):
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

    def __binary_to_string(self, binary_data, gzip_enabled=True):
        """Convert binary data to text.
        Parameters:
            binary_data(str): The binary data to be converted.
            gzip_enabled(bool, default=True): Whether to use gzip compression.
        Returns:
            The text.
        """
        byte_array = bytearray(int(binary_data[i:i+8], 2) for i in range(0, len(binary_data), 8))
        if gzip_enabled:
            try:
                byte_array = gzip.decompress(byte_array)
            except gzip.BadGzipFile:
                return "Decompression failed. Data could be corrupted."
        return byte_array.decode('utf-8', errors='ignore')

    def __demodulate(self, signal):
        """Demodulate the signal to binary data.
        Parameters:
            signal(np.array): The signal to demodulate.
        Returns:
            The binary data.
        """
        if self.modulation_mode == 1:
            return self.__bpsk_demodulate(signal)
        if self.modulation_mode == 2:
            return self.__mfsk_demodulate(signal, M=self.m_for_mfsk)
        return None

    def demodulate(self, signal, mode=2, compression_enabled=True):
        """Demodulate the signal to text.
        Parameters:
            signal(np.array): The signal to demodulate.
            mode(int, default=2): The demodulation mode. 
                                  Currently only supports BPSK (1), 64-FSK (2).
            compression_enabled(bool, default=True): Whether to decompress the 
                                                     binary data using GZIP.
        Returns:
            The text.
        """
        if mode == 1:
            binary_data = self.__bpsk_demodulate(signal)
        elif mode == 2:
            binary_data = self.__mfsk_demodulate(signal) # 64-FSK
        else:
            # Not Implemented
            return None
        return self.__binary_to_string(binary_data, gzip_enabled=compression_enabled)

    def stft(self, signal, window_size, step_size):
        """Performs Short-Time Fourier Transform on the signal.
        Parameters:
            signal (np.array): The signal to analyze.
            window_size (int): The size of each time window.
            step_size (int): The step size between windows.
        Returns:
            np.array: STFT matrix (time-frequency representation).
        """
        f, t, Zxx = scipy.signal.stft(signal, fs=self.sample_rate, window='hann',
                                    nperseg=window_size, noverlap=window_size - step_size)
        return f, t, np.abs(Zxx)

    def detect_stepped_chirp(self, signal, steps, duration):
        """Detects a stepped chirp in the signal.
        Parameters:
            signal (np.array): The signal to analyze.
            steps (list): The list of frequencies in the chirp.
            duration (float): The duration of the chirp in seconds.
        Returns:
            bool: Whether the chirp is detected.
        """
        window_size = int(self.sample_rate * duration / len(steps))
        step_size = window_size // 2
        f, t, Zxx = self.stft(signal, window_size, step_size)

        step_duration = duration / len(steps)
        tolerance = 20  # Frequency tolerance in Hz

        for i, expected_freq in enumerate(steps):
            # Time interval for this step
            start_time = i * step_duration
            end_time = (i + 1) * step_duration

            # Find the time indices corresponding to this interval
            time_indices = np.where((t >= start_time) & (t < end_time))[0]

            # Check if the expected frequency is present in these time indices
            freq_present = False
            for idx in time_indices:
                # Find peak frequency at this time index
                peak_freq = f[np.argmax(Zxx[:, idx])]
                # print(peak_freq)
                if np.abs(peak_freq - expected_freq) <= tolerance:
                    freq_present = True
                    break

            if not freq_present:
                return False  # Frequency step not detected

        return True  # All frequency steps detected

    def detect_preamble(self, signal):
        """Detects the preamble in the signal.
        Parameters:
            signal (np.array): The signal to analyze.
        Returns:
            bool: Whether the preamble is detected.
        """
        return self.detect_stepped_chirp(signal,
                                         self.start_chrip_seq,
                                         self.preamble_duration)
  
    def detect_postamble(self, signal):
        """Detects the postamble in the signal.
        Parameters:
            signal (np.array): The signal to analyze.
        Returns:
            bool: Whether the postamble is detected.
        """
        return self.detect_stepped_chirp(signal,
                                         self.end_chrip_seq,
                                         self.postamble_duration)
# Example usage
# demodulator = Demodulator()
# signal_64fsk = demodulator.read_from_wav("hello_world_64fsk.wav")
# decoded_text_64fsk = demodulator.demodulate(signal_64fsk, compression_enabled=False)
# print("64-FSK: \n", decoded_text_64fsk)

# signal_64fsk_gaussian = demodulator.read_from_wav("hello_world_64fsk_gaussian.wav")
# decoded_text_64fsk_gaussian = demodulator.demodulate(signal_64fsk_gaussian, 
#                                                      compression_enabled=False)
# print("64-FSK with Gaussian Noise: \n", decoded_text_64fsk_gaussian)

# signal_64fsk_gaussian_gzip = demodulator.read_from_wav("hello_world_64fsk_gaussian_gzip.wav")
# decoded_text_64fsk_gaussian_gzip = demodulator.demodulate(signal_64fsk_gaussian_gzip)
# print("64-FSK with Gaussian Noise and GZIP: \n", decoded_text_64fsk_gaussian_gzip)


# signal_bpsk = demodulator.read_from_wav("hello_world_bpsk.wav")
# decoded_text_bpsk = demodulator.demodulate(signal_bpsk, mode=1, compression_enabled=False)
# print("BPSK: \n", decoded_text_bpsk)

# signal_bpsk_gaussian = demodulator.read_from_wav("hello_world_bpsk_gaussian.wav")
# decoded_text_bpsk_gaussian = demodulator.demodulate(signal_bpsk_gaussian,
#                                                     mode=1,
#                                                     compression_enabled=False)
# print("BPSK with Gaussian Noise: \n", decoded_text_bpsk_gaussian)

# signal_bpsk_gaussian_gzip = demodulator.read_from_wav("hello_world_bpsk_gaussian_gzip.wav")
# decoded_text_bpsk_gaussian_gzip = demodulator.demodulate(signal_bpsk_gaussian_gzip, mode=1)
# print("BPSK with Gaussian Noise and GZIP: \n", decoded_text_bpsk_gaussian_gzip)
