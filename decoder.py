'''
The decoder module, responsible for decoding the carrier audio signal to text.
Liang Sidi, 2024
'''
import wave
import gzip
import numpy as np
import scipy.signal
from reedsolo import RSCodec, ReedSolomonError

class Decoder:
    """Class for decoding Hearme binary data to text.
    """
    def __init__(self, fec_enabled=True, fec_nsym=50,
                 compression_enabled=True):
        self.compression_enabled = compression_enabled
        self.fec_enabled = fec_enabled
        self.fec_nsym = fec_nsym
        if self.fec_enabled:
            self.rscodec = RSCodec(self.fec_nsym)

    def print_config(self):
        """Print the configuration of the decoder.
        """
        print("Hearme Decoder Module. Liang Sidi, 2024.")
        print(f"Compression Enabled: {self.compression_enabled}")
        print(f"FEC Enabled: {self.fec_enabled}")
        print(f"FEC Nsym: {self.fec_nsym}")

    def decode(self, binary_data, fec_enabled=True, compression_enabled=True):
        """Convert binary data to text.
        Parameters:
            binary_data(str): The binary data to be converted.
            fec_enabled(bool, default=True): Whether Forward Error Correction is used.
            compression_enabled(bool, default=True): Whether data is GZIP compressed.
        Returns:
            The text.
        """
        self.print_config()
        # print(binary_data)
        print("Msg Len: " + str(len(binary_data)))
        print(binary_data)
        byte_array = bytearray(int(binary_data[i:i+8], 2) for i in range(0, len(binary_data), 8))
        if fec_enabled:
            # Apply Reed-Solomon decoding
            try:
                byte_array, _, ecc = self.rscodec.decode(byte_array)
                print("ECC: " + str(ecc))
            except ReedSolomonError:
                return "Decoding failed: Too many errors to correct."
        if compression_enabled:
            # Decompress the data
            try:
                byte_array = gzip.decompress(byte_array)
            except gzip.BadGzipFile:
                return "Decompression failed. Data could be corrupted."
        return byte_array.decode('utf-8', errors='ignore')

    def get_message(self, binary_data):
        """Recover text from a signal.
        Protocol: "H#<text>##"
        Parameters:
            signal (np.array): The signal to recover text from.
        Returns:
            The text.
        """
        print(f"Length of binary data: {len(binary_data)}")
        print(binary_data)
        msg_header = "H#"
        msg_end = "##"
        msg_header_binary = ''.join(format(ord(char), '08b') for char in msg_header)
        msg_end_binary = ''.join(format(ord(char), '08b') for char in msg_end)
        sync_start_index = binary_data.find(msg_header_binary)
        sync_end_index = binary_data.find(msg_end_binary)
        if sync_start_index != -1:
            if sync_end_index == -1:
                return "Message incomplete."
            # Extract actual message starting from the end of sync symbol, 
            # until the end of the message
            message_binary = binary_data[sync_start_index + len(msg_header_binary):sync_end_index]
            print(message_binary)
            return self.decode(message_binary,
                               fec_enabled=self.fec_enabled,
                               compression_enabled=self.compression_enabled)
        return "No message detected."

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
        
    Methods:
        __init__(self, 
                carrier_freq=2000, 
                sample_rate=44100, 
                bit_duration=0.01,
                bandwidth=2000,
                modulation_mode=2,
                m_for_mfsk=64,
                fec_enabled=True,
                fec_nsym=10,
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
                 m_for_mfsk=64):
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
        """
        self.carrier_freq = carrier_freq
        self.sample_rate = sample_rate
        self.bit_duration = bit_duration
        self.bandwidth = bandwidth
        self.modulation_mode = modulation_mode
        # Check is M is a power of 2
        assert m_for_mfsk & (m_for_mfsk - 1) == 0 and m_for_mfsk != 0, "M must be a power of 2"
        self.m_for_mfsk = m_for_mfsk
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
 
    def print_config(self):
        """Print the configuration of the demodulator.
        """
        print("Hearme Demodulator Module. Liang Sidi, 2024.")
        # Print Parameters
        print(f"Carrier Frequency: {self.carrier_freq}")
        print(f"Sample Rate: {self.sample_rate}")
        print(f"Bit Duration: {self.bit_duration}")
        print(f"Bandwidth: {self.bandwidth}")
        print(f"Modulation Mode: {self.modulation_mode}")
        print(f"M for MFSK: {self.m_for_mfsk}")

    
    def get_binary(self, signal):
        """Demodulate the signal to binary data.
        Parameters:
            signal(np.array): The signal to demodulate.
        Returns:
            The binary data.
        """
        return self.__demodulate(signal)

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        """ A Butterworth bandpass filter.
        Parameters:
            lowcut (int): The lower cutoff frequency.
            highcut (int): The higher cutoff frequency.
            fs (int): The sampling rate.
            order (int): The order of the filter.
        Returns:
            The filter coefficients.
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = scipy.signal.butter(order, [low, high], btype='band')
        return b, a

    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        """Apply a bandpass filter to the data.
        Parameters:
            data (np.array): The data to filter.
            lowcut (int): The lower cutoff frequency.
            highcut (int): The higher cutoff frequency.
            fs (int): The sampling rate.
            order (int): The order of the filter.
        Returns:
            The filtered data.
        """
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = scipy.signal.lfilter(b, a, data)
        return y

    def __mfsk_demodulate(self, signal, M=64):
        """Demodulate the MFSK signal to binary data.
        Parameters: 
            signal (np.array): The signal to demodulate.
            M (int): The M in MFSK, representing the number of different frequencies. 
                     Must be a power of 2.
        Returns:
            The binary data.
        """
        # Filter the signal to the bandwidth
        # signal = self.bandpass_filter(signal, self.carrier_freq - self.bandwidth/2,
        #                               self.carrier_freq + self.bandwidth/2,
        #                               self.sample_rate)
        bits_per_symbol = int(np.log2(M))
        num_symbols = int(len(signal) / int(self.sample_rate * self.bit_duration))

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
            zero_padded_slice = np.pad(symbol_slice, (0, len(symbol_slice) * 3), 'constant')

            # Perform FFT
            fft_result = np.fft.fft(zero_padded_slice)
            freqs = np.fft.fftfreq(len(zero_padded_slice), 1 / self.sample_rate)

            # Focus on the positive frequencies only
            half_n = len(fft_result) // 2
            fft_result_positive = fft_result[:half_n]
            freqs_positive = freqs[:half_n]

            cut_high = self.carrier_freq + self.bandwidth/2 + 100 # Hz
            cut_low = self.carrier_freq - self.bandwidth/2 - 100 # Hz
            # Filter the FFT result to the bandwidth
            # fft_result_positive = fft_result_positive[(freqs_positive > cut_low)
            #                                           & (freqs_positive < cut_high)]

            # Find the frequency in symbol_freqs closest to the peak frequency in the FFT
            peak_freq = freqs_positive[np.argmax(np.abs(fft_result_positive))]
            # print(peak_freq)
            symbol = np.argmin(np.abs(symbol_freqs - peak_freq))
            binary_data += format(symbol, f'0{int(np.log2(M))}b')

        return binary_data
   
    def __mfsk_demodulate_new(self, signal, M=64):
        """Demodulate the MFSK signal to binary data using STFT.
        Parameters:
            signal (np.array): The signal to demodulate.
            M (int): The M in MFSK, representing the number of different frequencies.
        Returns:
            The binary data.
        """
        bits_per_symbol = int(np.log2(M))
        symbol_duration = int(self.sample_rate * self.bit_duration)
        symbol_freqs = np.linspace(self.carrier_freq - self.bandwidth/2,
                                   self.carrier_freq + self.bandwidth/2, M)
        num_symbols = len(signal) // int(self.sample_rate * self.bit_duration)

        binary_data = ""
        for i in range(num_symbols):
            start = i * int(self.sample_rate * self.bit_duration)
            end = start + int(self.sample_rate * self.bit_duration)
            symbol_slice = signal[start:end]
            
            # Perform STFT
            f, t, Zxx = scipy.signal.stft(symbol_slice, fs=self.sample_rate, nperseg=symbol_duration)
            
            # Find the peak frequency for each time bin in the STFT result
            peak_freqs = f[np.argmax(np.abs(Zxx), axis=0)]
            
            # Average the peak frequencies over the symbol duration to get the dominant frequency
            dominant_freq = np.mean(peak_freqs)
            # print(dominant_freq)
            
            # Find the nearest carrier frequency
            symbol_index = np.argmin(np.abs(symbol_freqs - dominant_freq))
            binary_data += format(symbol_index, f'0{bits_per_symbol}b')

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

    def __demodulate(self, signal):
        """Demodulate the signal to binary data.
        Parameters:
            signal(np.array): The signal to demodulate.
        Returns:
            The demodulated binary data.
        """
        self.print_config()
        if self.modulation_mode == 1:
            return self.__bpsk_demodulate(signal)
        if self.modulation_mode == 2:
            return self.__mfsk_demodulate(signal, M=self.m_for_mfsk)
        return None

    def demodulate(self, signal, mode=2):
        """Demodulate the signal to text.
        Parameters:
            signal(np.array): The signal to demodulate.
            mode(int, default=2): The demodulation mode. 
                                  Currently only supports BPSK (1), 64-FSK (2).
        Returns:
            The demodulated binary data.
        """
        self.print_config()
        if mode == 1:
            binary_data = self.__bpsk_demodulate(signal)
        elif mode == 2:
            binary_data = self.__mfsk_demodulate(signal) # 64-FSK
        else:
            # Not Implemented
            return None
        return binary_data

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
demodulator = Demodulator()
decoder = Decoder()
demodulator.bit_duration = 0.05  # 100 ms bit duration
demodulator.carrier_freq = 16000  # 16 kHz carrier frequency
signal_64fsk = demodulator.read_from_wav("hello_world_64fsk.wav")
demodulated_binary_64fsk = demodulator.demodulate(signal_64fsk)
decoded_text_64fsk = decoder.decode(demodulated_binary_64fsk, 
                                    compression_enabled=False, 
                                    fec_enabled=False)
print("64-FSK: \n", decoded_text_64fsk)

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
