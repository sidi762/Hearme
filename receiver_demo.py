import hearme
# Example usage
demodulator = hearme.Demodulator()  # Initialize demodulator with appropriate settings
# Configure demodulator settings
demodulator.bit_duration = 0.05  # 500 ms bit duration
demodulator.carrier_freq = 16000  # 16 kHz carrier frequency
demodulator.bandwidth = 4400  # 8.8 kHz bandwidth
demodulator.sample_rate = 44100  # Audio sample rate
demodulator.m_for_mfsk = 64  # MFSK modulation order
decoder = hearme.Decoder()
# Configure decoder settings
decoder.compression_enabled = False
real_time_demod = hearme.HearmeDetector(demodulator, decoder)
real_time_demod.start()
