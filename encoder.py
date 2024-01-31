"""The encoder module, responsible for encoding text to carrier audio signals.
Liang Sidi, 2024
"""
import wave
import gzip
import numpy as np

class Modulator:
    """Class for modulating text to carrier audio signals.
    Attributes:
        carrier_freq(int): The frequency of the carrier signal in Hz.
        sample_rate(int): The sampling rate in Hz.
        bit_duration(float): The duration of each bit in seconds.
        bandwidth(int): The bandwidth of the signal in Hz.
        modulation_mode(int): The modulation mode.
                              Currently only supports BPSK (1), MFSK (2).
        m_for_mfsk(int): The M in MFSK, default to 64-FSK. Must be a power of 2.
        compression_enabled(bool): Whether to compress the binary data using GZIP.
        
    Methods:
        __init__(self, 
                carrier_freq=1000, 
                sample_rate=44100, 
                bit_duration=0.01,
                bandwidth=4400,
                modulation_mode=2,
                m_for_mfsk=64,
                compression_enabled=True): Initializes the modulator.
        generate_signal(self, text): Generates the acoustic signal for the given text.
        __string_to_binary(self, text): Converts text to binary data.
        __bpsk_modulate(self, binary_data): Modulates binary data to BPSK signal.
        __mfsk_modulate(self, binary_data, M=64): Modulates binary data to MFSK signal.
        __modulate(self, text, mode=1): Modulates text to carrier audio signal.
        modulate(self, text, mode=1, compression_enabled=True): Modulates text to carrier 
                                        audio signal. Does not follow class configuration.
        __save_to_wav(self, signal, filename="output.wav"): Saves the signal to a WAV file.    
    """
    def __init__(self, carrier_freq=8800, sample_rate=44100,
                 bit_duration=0.01, bandwidth=4400, modulation_mode=2,
                 m_for_mfsk=64, compression_enabled=True):
        """Initializes the modulator.
        Parameters:
            carrier_freq(int, default=8800): The frequency of the carrier signal in Hz.
            sample_rate(int, default=44100): The sampling rate in Hz.
            bit_duration(float, default=0.01): The duration of each bit in seconds.
            bandwidth(int, default=4400): The bandwidth of the signal in Hz.
            modulation_mode(int, default=2): The modulation mode.
                                             Currently only supports BPSK (1), MFSK (2).
            m_for_mfsk(int, default=64): The M in MFSK, default to 64-FSK. Must be a power of 2.
            compression_enabled(bool, default=True): Whether to compress the binary data using GZIP.
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
        # Chrip signal frequencies for preamble and postamble
        # C major scale frequencies (C4 to C6)
        self.start_chrip_seq = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25,
                         587.33, 659.25, 698.46, 783.99, 880.00, 987.77, 1046.50]
        self.end_chrip_seq = self.start_chrip_seq[::-1]
        self.preamble_duration = 1
        self.postamble_duration = 1

    def generate_signal(self, text):
        """Generates the signal from text.
        Protocol: Chirp + 0.5s + "HMMSG##<text>" + 0.5s + Chirp + 1s
        Parameters:
            text(str): The text to be modulated.
        Returns:
            The signal.
        """
        stepped_chirp_start = self.generate_stepped_chirp(self.preamble_duration,
                                                          self.start_chrip_seq)
        stepped_chirp_end = self.generate_stepped_chirp(self.postamble_duration,
                                                        self.end_chrip_seq)
        text = f"HMMSG##{text}"
        binary_data = self.__string_to_binary(text, gzip_enabled=self.compression_enabled)
        print(binary_data)
        _, modulated_signal = self.__modulate(binary_data)
        signal = np.concatenate((stepped_chirp_start, np.zeros(int(self.sample_rate * 0.5)),
                                    modulated_signal, np.zeros(int(self.sample_rate * 0.5)),
                                    stepped_chirp_end, np.zeros(self.sample_rate)))
        return signal

    def generate_chirp(self, start_freq, end_freq, duration):
        """Generates a chirp signal.
        Parameters:
            start_freq (float): The starting frequency of the chirp.
            end_freq (float): The ending frequency of the chirp.
            duration (float): The duration of the chirp in seconds.
        Returns:
            numpy.array: The chirp signal.
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        chirp_signal = np.sin(2 * np.pi * t * np.linspace(start_freq, end_freq, t.size))
        return chirp_signal

    def generate_stepped_chirp(self, duration, steps):
        """Generates a stepped chirp signal.
        Parameters:
            start_freq (float): The starting frequency of the chirp.
            end_freq (float): The ending frequency of the chirp.
            duration (float): The total duration of the chirp in seconds.
            steps (list): List of frequencies for each step (musical notes).
        Returns:
            numpy.array: The chirp signal.
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        chirp_signal = np.zeros_like(t)

        step_duration = duration / len(steps)
        for i, freq in enumerate(steps):
            start_idx = int(i * step_duration * self.sample_rate)
            end_idx = int((i + 1) * step_duration * self.sample_rate)
            step_t = t[start_idx:end_idx]
            chirp_signal[start_idx:end_idx] = np.sin(2 * np.pi * freq * step_t)

        return chirp_signal

    def __string_to_binary(self, text, gzip_enabled=True):
        """Converts text to binary data.
        Parameters:
            text(str): The text to be converted.
            gzip_enabled(bool, default=True): Whether to compress the binary data using GZIP.
        Returns:
            The binary data.
        """
        binary_data = ''.join(format(byte, '08b') for byte in text.encode('utf-8'))
        # print(binary_data)
        if gzip_enabled:
            data_bytes = int(binary_data, 2).to_bytes((len(binary_data) + 7) // 8, byteorder='big')
            compressed_data = gzip.compress(data_bytes)
            binary_data = ''.join(format(byte, '08b') for byte in compressed_data)
            # print(binary_data)
        return binary_data

    def __mfsk_modulate(self, binary_data, M=64):
        """Modulates binary data to MFSK signal.
        Parameters:
            binary_data(str): The binary data to be modulated.
            M(int, default=64): The M in MFSK, representing the number of different frequencies.
        Returns:
            The modulated signal.
        """
        # Number of bits per symbol (e.g., M=4 means 2 bits per symbol for 4-FSK)
        bits_per_symbol = int(np.log2(M))

        # Calculate the number of symbols
        num_symbols = int(len(binary_data) / bits_per_symbol)

        # Frequencies for each symbol
        # symbol_freqs = np.linspace(self.carrier_freq - M*440/2, self.carrier_freq + M*440/2, M)
        symbol_freqs = np.linspace(self.carrier_freq - self.bandwidth/2,
                                   self.carrier_freq + self.bandwidth/2,
                                   M)

        # Create the signal array
        num_samples = int(num_symbols * self.sample_rate * self.bit_duration)
        signal = np.zeros(num_samples)

        for i in range(num_symbols):
            # Extract the bits for this symbol
            symbol_bits = binary_data[i*bits_per_symbol:(i+1)*bits_per_symbol]

            # Convert bits to a decimal symbol
            symbol = int(symbol_bits, 2)

            # Frequency for this symbol
            freq = symbol_freqs[symbol]
            # print(freq)
            # Generate the signal for this symbol
            start = int(i * self.sample_rate * self.bit_duration)
            end = int((i + 1) * self.sample_rate * self.bit_duration)
            t = np.linspace(start / self.sample_rate,
                            end / self.sample_rate,
                            end - start,
                            endpoint=False)
            signal[start:end] = np.cos(2 * np.pi * freq * t)

        return t, signal

    def __bpsk_modulate(self, binary_data):
        """Modulates binary data to BPSK signal.
        Parameters:
            binary_data(str): The binary data to be modulated.
        Returns:
            The modulated signal.
        """
        num_samples = int(len(binary_data) * self.sample_rate * self.bit_duration)
        signal = np.zeros(num_samples)
        for i, bit in enumerate(binary_data):
            phase = np.pi if bit == '1' else 0
            start = int(i * self.sample_rate * self.bit_duration)
            end = int((i + 1) * self.sample_rate * self.bit_duration)
            t = np.linspace(start / self.sample_rate,
                            end / self.sample_rate,
                            end - start,
                            endpoint=False)
            signal[start:end] = np.cos(2 * np.pi * self.carrier_freq * t + phase)
        return t, signal

    def __modulate(self, binary_data):
        """Modulates binary data to carrier audio signal.
        Parameters:
            binary_data(str): The binary data to be modulated.
        Returns:
            The modulated signal.
        """
        if self.modulation_mode == 1:
            return self.__bpsk_modulate(binary_data)
        if self.modulation_mode == 2:
            return self.__mfsk_modulate(binary_data, M=self.m_for_mfsk)
        return None

    def modulate(self, text, mode=2, compression_enabled=True):
        """Modulates text to carrier audio signal.
        Parameters: 
            text(str): The text to be modulated.
            mode(int, default=2): The modulation mode. 
                                  Currently only supports BPSK (1), 64-FSK (2).
            compression_enabled(bool, default=True): Whether to compress the binary data using GZIP.
        Returns:
            The modulated signal.
        """
        binary_data = self.__string_to_binary(text, gzip_enabled=compression_enabled)
        if mode == 1:
            return self.__bpsk_modulate(binary_data)
        elif mode == 2:
            return self.__mfsk_modulate(binary_data) # 64-FSK
        else:
            return None

    def save_to_wav(self, signal, filename="output.wav", add_gaussian_noise=False):
        """Save the signal to a WAV file.
        Parameters:
            signal: The signal to save.
            filename(str, default="output.wav"): The name of the file to save to.
            add_gaussian_noise(bool, default=False): Whether to add Gaussian 
                                                     noise (for testing) to the signal.
        """
        if add_gaussian_noise:
            noise = np.random.normal(0, 1, len(signal)).astype(np.int16)
            signal += noise
        # Normalizing the signal to fit in the 16-bit range
        normalized_signal = np.int16((signal / signal.max()) * 32767)


        # Open a WAV file for writing
        with wave.open(filename, 'w') as wav_file:
            # Set the parameters for the WAV file
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 16 bits per sample
            wav_file.setframerate(self.sample_rate)

            # Write the normalized signal to the WAV file
            wav_file.writeframes(normalized_signal.tobytes())

        print(f"File saved as {filename}")

    

# Example usage
modulator = Modulator()
modulator.compression_enabled = False
modulator.bit_duration = 0.1
# text_to_encode = "The quick brown fox jumps over the lazy dog."
TEXT_TO_ENCODE = "The quick brown fox jumps over the lazy dog. \n\
                  天地玄黄，宇宙洪荒，日月盈仄，辰宿列张，\n\
                  寒来暑往，秋收冬藏，闰余成岁，律吕调阳。"
generated_signal = modulator.generate_signal(TEXT_TO_ENCODE)
modulator.save_to_wav(generated_signal, "hello_world_stream_test_64fsk.wav",
                      add_gaussian_noise=False)

# For testing
_, signal = modulator.modulate(TEXT_TO_ENCODE, mode=2, compression_enabled=False)
# play the signal
# import sounddevice as sd
# sd.play(signal, modulator.sample_rate, blocking=True)

modulator.save_to_wav(signal, "hello_world_64fsk.wav", add_gaussian_noise=False)
modulator.save_to_wav(signal, "hello_world_64fsk_gaussian.wav", add_gaussian_noise=True)
_, signal_gz = modulator.modulate(TEXT_TO_ENCODE, mode=2, compression_enabled=True)
modulator.save_to_wav(signal_gz, "hello_world_64fsk_gzip.wav", add_gaussian_noise=False)
modulator.save_to_wav(signal_gz, "hello_world_64fsk_gaussian_gzip.wav", add_gaussian_noise=True)


_, signal_bpsk = modulator.modulate(TEXT_TO_ENCODE, mode=1, compression_enabled=False)
modulator.save_to_wav(signal_bpsk, "hello_world_bpsk.wav")
modulator.save_to_wav(signal_bpsk, "hello_world_bpsk_gaussian.wav", add_gaussian_noise=True)
_, signal_bpsk_gz = modulator.modulate(TEXT_TO_ENCODE, mode=1, compression_enabled=True)
modulator.save_to_wav(signal_bpsk_gz, "hello_world_bpsk_gzip.wav", add_gaussian_noise=False)
modulator.save_to_wav(signal_bpsk_gz, "hello_world_bpsk_gaussian_gzip.wav", add_gaussian_noise=True)
