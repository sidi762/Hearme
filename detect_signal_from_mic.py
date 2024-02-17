import pyaudio
import numpy as np
import scipy
from decoder import Demodulator, Decoder

class HearmeDetector:
    def __init__(self, demodulator, decoder):
        self.demodulator = demodulator
        self.decoder = decoder
        self.decoder.compression_enabled = False
        self.demodulator.bit_duration = 0.05  # 500 ms bit duration
        self.demodulator.carrier_freq = 16000  # 16 kHz carrier frequency
        self.demodulator.bandwidth = 4400  # 8.8 kHz bandwidth
        self.chunk_size = 1024 # Number of audio samples per read
        self.demodulator.sample_rate = 44100  # Audio sample rate
        self.demodulator.m_for_mfsk = 64  # MFSK modulation order
        self.format = pyaudio.paInt16  # Audio format (16-bit)
        self.channels = 1  # Number of audio channels (mono)
        self.rate = demodulator.sample_rate  # Sampling rate
        # 1.5 second window size in samples
        self.window_size = int(1.5 * self.rate / self.chunk_size) * self.chunk_size
        self.sliding_window = np.zeros(self.window_size, dtype=np.int16)  # Sliding window buffer
        self.state = 'listening'
        self.message_data = np.array([])

        # Define filter characteristics
        self.nyquist = 0.5 * self.rate
        low = (self.demodulator.carrier_freq - self.demodulator.bandwidth / 2) / self.nyquist
        high = (self.demodulator.carrier_freq + self.demodulator.bandwidth / 2) / self.nyquist

        # Design a Butterworth bandpass filter
        self.b, self.a = scipy.signal.butter(4, [low, high], btype='band')
        print("Hearme Listener Module. Liang Sidi, 2024.")

    def apply_filter(self, data):
        """Apply the designed bandpass filter to the data."""
        return scipy.signal.lfilter(self.b, self.a, data)

    def start(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=self.format, channels=self.channels,
                            rate=self.rate, input=True,
                            frames_per_buffer=self.chunk_size)
        print("Listening...")
        frames = [] # A python-list of chunks(numpy.ndarray)
        try:
            while True:
                data = stream.read(self.chunk_size)
                np_data = np.frombuffer(data, dtype=np.int16)
                filtered_data = self.apply_filter(np_data)
                self.update_buffer(np_data)
                if self.state == 'listening':
                    if self.demodulator.detect_preamble(self.sliding_window):
                        frames.append(np_data)
                        self.state = 'preamble_detected'
                        print("Preamble Detected")
                elif self.state == 'preamble_detected':
                    frames.append(filtered_data)
                    if self.demodulator.detect_postamble(self.sliding_window):
                        self.state = 'postamble_detected'
                        print("Postamble Detected")
                        stream.stop_stream()
                        # Convert frames to nparray
                        self.message_data = np.concatenate(frames)
                        binary = self.demodulator.get_binary(self.message_data)
                        text = self.decoder.get_message(binary)
                        print("Detected Text:", text)
                        self.clear_message_buffer()  # Clear the buffer after processing
                        frames = []
                        self.state = 'listening'
                        print("Listening...")
                        stream.start_stream()

        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

    def update_buffer(self, new_data):
        """Update the sliding window buffer with new data."""
        # Shift the existing data to the left
        self.sliding_window[:-len(new_data)] = self.sliding_window[len(new_data):]
        # Append new data to the end of the window
        self.sliding_window[-len(new_data):] = new_data

    def clear_message_buffer(self):
        self.message_data = np.array([])




# Example usage
demodulator = Demodulator()  # Initialize demodulator with appropriate settings
decoder = Decoder()
real_time_demod = HearmeDetector(demodulator, decoder)
real_time_demod.start()
