"""
Implements signal detection logic to identify the presence 
of sound signals using the microphone (with pyaudio). 
This module includes methods for preamble detection, 
signal strength analysis, and synchronization to facilitate 
accurate decoding.
Liang Sidi, 2024
"""

import pyaudio
import numpy as np
import scipy

class HearmeDetector:
    """
    A class that represents a Hearme detector.

    Attributes:
        demodulator (Demodulator): The demodulator object used for demodulation.
        decoder (Decoder): The decoder object used for decoding.
        chunk_size (int): Number of audio samples per read.
        format (int): Audio format (16-bit).
        channels (int): Number of audio channels (mono).
        rate (int): Sampling rate.
        window_size (int): Size of the sliding window in samples.
        sliding_window (numpy.ndarray): Sliding window buffer.
        state (str): Current state of the detector.
        message_data (numpy.ndarray): Buffer for storing message data.

    Methods:
        __init__(self, demodulator, decoder): Initializes the HearmeDetector object.
        apply_filter(self, data): Applies the bandpass filter to the data.
        start(self): Starts the detection process.
        update_buffer(self, new_data): Updates the sliding window buffer with new data.
        clear_message_buffer(self): Clears the message data buffer.
    """

    def __init__(self, demodulator, decoder):
        """
        Initializes the HearmeDetector object.

        Args:
            demodulator (Demodulator): The demodulator object used for demodulation.
            decoder (Decoder): The decoder object used for decoding.
        """
        self.demodulator = demodulator
        self.decoder = decoder
        self.chunk_size = 1024 # Number of audio samples per read
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
        """
        Apply the designed bandpass filter to the data.

        Args:
            data (numpy.ndarray): The input audio data.

        Returns:
            numpy.ndarray: The filtered audio data.
        """
        return scipy.signal.lfilter(self.b, self.a, data)

    def start(self):
        """
        Start the detection process.
        """
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
        """
        Update the sliding window buffer with new data.

        Args:
            new_data (numpy.ndarray): The new data to be added to the buffer.
        """
        # Shift the existing data to the left
        self.sliding_window[:-len(new_data)] = self.sliding_window[len(new_data):]
        # Append new data to the end of the window
        self.sliding_window[-len(new_data):] = new_data

    def clear_message_buffer(self):
        """
        Clear the message data buffer.
        """
        self.message_data = np.array([])
