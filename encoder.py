"""The encoder module, responsible for encoding text to carrier audio signals.
Liang Sidi, 2024
"""
import wave
import gzip
import numpy as np
from reedsolo import RSCodec

class Encoder:
    """Class for encoding text for transmission.
    Parameters:
            fec_enabled(bool, default=True): Whether to enable Reed-Solomon 
                                             forward error correction.
            fec_nsym(int, default=10): The number of symbols for Reed-Solomon FEC.
            compression_enabled(bool, default=True): Whether to compress the binary data using GZIP.
    """
    def __init__(self,fec_enabled=True, fec_nsym=50,
                 compression_enabled=True):
        self.fec_enabled = fec_enabled
        self.fec_nsym = fec_nsym
        if fec_enabled:
            self.rscodec = RSCodec(self.fec_nsym)
        self.compression_enabled = compression_enabled
        
    def print_config(self):
        """Prints the configuration of the encoder.
        """
        print("Hearme Encoder Module. Liang Sidi, 2024.")
        print(f"FEC Enabled: {self.fec_enabled}")
        print(f"FEC Nsym: {self.fec_nsym}")
        print(f"Compression Enabled: {self.compression_enabled}")
        
    def encode(self, text):
        """Encodes the text for transmission.
        Parameters:
            text(str): The text to be encoded.
        Returns:
            The encoded text.
        """
        binary_data = self.generate_binary(text,
                                           gzip_enabled=self.compression_enabled,
                                           fec_enabled=self.fec_enabled)
        return binary_data

    def generate_binary(self, text, gzip_enabled=True, fec_enabled=True):
        """Converts text to binary data.
        Parameters:
            text(str): The text to be converted.
            gzip_enabled(bool, default=True): Whether to compress the binary data using GZIP.
            fec_enabled(bool, default=True): Whether to apply Reed-Solomon forward error correction.
        Returns:
            The binary data.
        """
        self.print_config()
        data_bytes = text.encode('utf-8')
        binary_data = ''.join(format(byte, '08b') for byte in data_bytes)
        print("Raw msg: " + binary_data)
        print("___________________________________________________________________________")
        raw_msg_len = len(binary_data)
        print(f"Raw msg len: {raw_msg_len}")
        if gzip_enabled:
            # Compress the binary data using GZIP
            data_bytes = gzip.compress(data_bytes)
            compressed_msg_len = len(data_bytes)*8 # in bits
            print(f"Compressed msg len: {compressed_msg_len}")
        if fec_enabled:
            # Apply Reed-Solomon forward error correction
            data_bytes = self.rscodec.encode(data_bytes)
        binary_data = ''.join(format(byte, '08b') for byte in data_bytes)
        encoded_len = len(binary_data)
        print(f"Encoded msg len: {encoded_len}")
        print("Encoded msg: " + binary_data)
        return binary_data

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
        start_chrip_seq(list): Chrip signal frequencies for preamble.
        end_chrip_seq(list): Chrip signal frequencies for postamble.
        preamble_duration(float): The duration of the preamble in seconds.
        postamble_duration(float): The duration of the postamble in seconds.
        
        
    Methods:
        __init__(self, 
                carrier_freq=1000, 
                sample_rate=44100, 
                bit_duration=0.01,
                bandwidth=4400,
                modulation_mode=2,
                m_for_mfsk=64,
                fec_enabled=True,
                fec_nsym=10,
                compression_enabled=True): Initializes the modulator.
        generate_signal(self, text): Generates the acoustic signal for the given text.
        __bpsk_modulate(self, binary_data): Modulates binary data to BPSK signal.
        __mfsk_modulate(self, binary_data, M=64): Modulates binary data to MFSK signal.
        __modulate(self, text, mode=1): Modulates text to carrier audio signal.
        modulate(self, text, mode=1, compression_enabled=True): Modulates text to carrier 
                                        audio signal. Does not follow class configuration.
        __save_to_wav(self, signal, filename="output.wav"): Saves the signal to a WAV file.    
    """
    def __init__(self, carrier_freq=8800, sample_rate=44100,
                 bit_duration=0.01, bandwidth=4400, modulation_mode=2,
                 m_for_mfsk=64):
        """Initializes the modulator.
        Parameters:
            carrier_freq(int, default=8800): The frequency of the carrier signal in Hz.
            sample_rate(int, default=44100): The sampling rate in Hz.
            bit_duration(float, default=0.01): The duration of each bit in seconds.
            bandwidth(int, default=4400): The bandwidth of the signal in Hz.
            modulation_mode(int, default=2): The modulation mode.
                                             Currently only supports BPSK (1), MFSK (2).
            m_for_mfsk(int, default=64): The M in MFSK, default to 64-FSK. Must be a power of 2.
        """
        self.carrier_freq = carrier_freq
        self.sample_rate = sample_rate
        self.bit_duration = bit_duration
        self.bandwidth = bandwidth
        self.modulation_mode = modulation_mode
        # Check is M is a power of 2
        assert m_for_mfsk & (m_for_mfsk - 1) == 0 and m_for_mfsk != 0, "M must be a power of 2"
        self.m_for_mfsk = m_for_mfsk
        # Chrip signal frequencies for preamble and postamble
        # C major scale frequencies (C4 to C6)
        self.start_chrip_seq = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25,
                         587.33, 659.25, 698.46, 783.99, 880.00, 987.77, 1046.50]
        self.end_chrip_seq = self.start_chrip_seq[::-1]
        self.preamble_duration = 1
        self.postamble_duration = 1


    def print_config(self):
        """ Prints the configuration of the modulator.
        """
        print("Hearme Modulator Module. Liang Sidi, 2024.")
        print("Generating signal...")
        # Print Parameters
        print(f"Carrier Frequency: {self.carrier_freq}")
        print(f"Sample Rate: {self.sample_rate}")
        print(f"Bit Duration: {self.bit_duration}")
        print(f"Bandwidth: {self.bandwidth}")
        print(f"Modulation Mode: {self.modulation_mode}")
        print(f"M for MFSK: {self.m_for_mfsk}")

    def generate_signal(self, binary_data):
        """Generates the signal from binary data.
        Protocol: Chrip + 0.5s + H# + Data + ## + Chrip + 0.5s
        Parameters:
            text(str): The text to be modulated.
        Returns:
            The signal.
        """
        stepped_chirp_start = self.generate_stepped_chirp(self.preamble_duration,
                                                          self.start_chrip_seq)
        stepped_chirp_end = self.generate_stepped_chirp(self.postamble_duration,
                                                        self.end_chrip_seq)
        # print(binary_data)
        msg_header = "H#" # Header for the message, len=16 bits
        msg_end = "##" # End of message, len=16 bits
        msg_header_binary = ''.join(format(ord(char), '08b') for char in msg_header)
        msg_end_binary = ''.join(format(ord(char), '08b') for char in msg_end)
        binary_data = msg_header_binary + binary_data + msg_end_binary
        print(f"Sent data length: {len(binary_data)}")
        print(f"Sent data: {binary_data}")
        _, modulated_signal = self.__modulate(binary_data)
        signal = np.concatenate((stepped_chirp_start, np.zeros(int(self.sample_rate * 0.5)),
                                 modulated_signal,
                                 np.zeros(int(self.sample_rate * self.bit_duration)),
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

        # Calculate the number of symbols (pad the trailing bits with zeros if necessary)
        binary_data += '0' * (bits_per_symbol - len(binary_data) % bits_per_symbol)
        num_symbols = len(binary_data) // bits_per_symbol
        print(f"Num symbols: {num_symbols}")
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
        self.print_config()
        if self.modulation_mode == 1:
            return self.__bpsk_modulate(binary_data)
        if self.modulation_mode == 2:
            return self.__mfsk_modulate(binary_data, M=self.m_for_mfsk)
        return None

    def modulate(self, binary_data, mode=2):
        """Modulates text to carrier audio signal.
        Parameters: 
            text(str): The text to be modulated.
            mode(int, default=2): The modulation mode. 
                                  Currently only supports BPSK (1), 64-FSK (2).
        Returns:
            The modulated signal.
        """
        self.print_config()
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
encoder = Encoder()
encoder.compression_enabled = False
modulator.carrier_freq = 16000
modulator.bandwidth = 4400
modulator.m_for_mfsk = 64
modulator.sample_rate = 44100
modulator.bit_duration = 0.05
# text_to_encode = "The quick brown fox jumps over the lazy dog."
# TEXT_TO_ENCODE = "The quick brown fox jumps over the lazy dog. \n\
#                   天地玄黄，宇宙洪荒，日月盈仄，辰宿列张，\n\
#                   寒来暑往，秋收冬藏，闰馀成岁，律吕调阳。"
TEXT_TO_ENCODE = "Hello, World!"
ENCODED_DATA = encoder.encode(TEXT_TO_ENCODE)
generated_signal = modulator.generate_signal(ENCODED_DATA)
modulator.save_to_wav(generated_signal, "hello_world_stream_test_64fsk.wav",
                      add_gaussian_noise=False)

# # For testing
# ENCODED_DATA = encoder.generate_binary(TEXT_TO_ENCODE, gzip_enabled=False, fec_enabled=False)
# _, signal = modulator.modulate(ENCODED_DATA, mode=2)
# play the signal
# # import sounddevice as sd
# # sd.play(signal, modulator.sample_rate, blocking=True)
# modulator.save_to_wav(signal, "hello_world_64fsk.wav", add_gaussian_noise=False)

# _, signal_bpsk = modulator.modulate(ENCODED_DATA, mode=1)
# modulator.save_to_wav(signal_bpsk, "hello_world_bpsk.wav")
# modulator.save_to_wav(signal_bpsk, "hello_world_bpsk_gaussian.wav", add_gaussian_noise=True)

# modulator.save_to_wav(signal, "hello_world_64fsk_gaussian.wav", add_gaussian_noise=True)
# ENCODED_DATA = encoder.generate_binary(TEXT_TO_ENCODE, gzip_enabled=True, fec_enabled=True)
# _, signal_gz = modulator.modulate(TEXT_TO_ENCODE, mode=2)
# modulator.save_to_wav(signal_gz, "hello_world_64fsk_gzip.wav", add_gaussian_noise=False)
# modulator.save_to_wav(signal_gz, "hello_world_64fsk_gaussian_gzip.wav", add_gaussian_noise=True)

# _, signal_bpsk_gz = modulator.modulate(ENCODED_DATA, mode=1)
# modulator.save_to_wav(signal_bpsk_gz, "hello_world_bpsk_gzip.wav", add_gaussian_noise=False)
# modulator.save_to_wav(signal_bpsk_gz, "hello_world_bpsk_gaussian_gzip.wav", add_gaussian_noise=True)
