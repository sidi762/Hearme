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
        
    Methods:
        __init__(self, 
                carrier_freq=1000, 
                sample_rate=44100, 
                bit_duration=0.1): Initializes the modulator.
        string_to_binary(self, text): Converts text to binary data.
        bpsk_modulate(self, binary_data): Modulates binary data to BPSK signal.
        modulate(self, text, mode=1): Modulates text to carrier audio signal.
        save_to_wav(self, signal, filename="output.wav"): Saves the signal to a WAV file.    
    """
    def __init__(self, carrier_freq=2200, sample_rate=44100, bit_duration=0.1, bandwidth=2200):
        """Initializes the modulator.
        Parameters:
            carrier_freq(int, default=2000): The frequency of the carrier signal in Hz.
            sample_rate(int, default=44100): The sampling rate in Hz.
            bit_duration(float, default=0.1): The duration of each bit in seconds.
            bandwidth(int, default=2200): The bandwidth of the signal in Hz.
        """
        self.carrier_freq = carrier_freq
        self.sample_rate = sample_rate
        self.bit_duration = bit_duration
        self.bandwidth = bandwidth
    
    def string_to_binary(self, text, gzip_enabled=True):
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

    def bpsk_modulate(self, binary_data):
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

    def modulate(self, text, mode=1):
        """Modulates text to carrier audio signal.
        Parameters: 
            text(str): The text to be modulated.
            mode(int, default=1): The modulation mode. Currently only supports BPSK (1).
        Returns:
            The modulated signal.
        """
        if mode == 1:
            binary_data = self.string_to_binary(text)
            return self.bpsk_modulate(binary_data)
        else:
            return None

    # Add more modulation methods here in future.
    def save_to_wav(self, signal, filename="output.wav"):
        """Save the signal to a WAV file.
        Parameters:
            signal: The signal to save.
            filename(str, default="output.wav"): The name of the file to save to.
        """
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
t, signal = modulator.modulate("The quick brown fox jumps over the lazy dog.")

modulator.save_to_wav(signal, "hello_world_bpsk.wav")

# Displaying the first 10 bits of the signal
# Number of bits to display in the plot
num_bits_to_display = 10

# Calculate the number of samples to display
samples_to_display = int(num_bits_to_display * modulator.bit_duration * modulator.sample_rate)

# Adjust the time and signal arrays to only include the desired number of samples
t_short = t[0:1024]
signal_short = signal[0:1024]


# Plotting the shortened signal
plt.figure(figsize=(15, 4))
plt.plot(t_short, signal_short)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title(f'BPSK Modulated Signal of "Hello World" (First {num_bits_to_display} bits)')
plt.grid(True)
plt.show()
