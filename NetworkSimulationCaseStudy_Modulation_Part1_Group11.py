import numpy as np
import matplotlib.pyplot as plt

# Part 1 - Modulation Technique

def binary_to_ask(binary_data, carrier_freq, sampling_rate):
    time = np.arange(0, len(binary_data) / sampling_rate, 1 / sampling_rate)
    ask_signal = []
    for bit in binary_data:
        if bit == '0':
            ask_signal.extend(np.sin(2 * np.pi * carrier_freq * time))
        elif bit == '1':
            ask_signal.extend(np.sin(4 * np.pi * carrier_freq * time))
    return ask_signal

def binary_to_fsk(binary_data, freq_logic_0, freq_logic_1, sampling_rate):
    time = np.arange(0, len(binary_data) / sampling_rate, 1 / sampling_rate)
    fsk_signal = []
    for bit in binary_data:
        if bit == '0':
            fsk_signal.extend(np.sin(2 * np.pi * freq_logic_0 * time))
        elif bit == '1':
            fsk_signal.extend(np.sin(2 * np.pi * freq_logic_1 * time))
    return fsk_signal

# Generate random 16-bit binary data
binary_data = ''.join([str(np.random.randint(0, 2)) for _ in range(16)])

# Parameters
sampling_rate = 1000  # Hz
carrier_freq_ask = 50  # Hz
freq_logic_0_fsk = 25  # Hz
freq_logic_1_fsk = 50  # Hz

# Modulation
ask_modulated_signal = binary_to_ask(binary_data, carrier_freq_ask, sampling_rate)
fsk_modulated_signal = binary_to_fsk(binary_data, freq_logic_0_fsk, freq_logic_1_fsk, sampling_rate)

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(np.arange(len(ask_modulated_signal)) / sampling_rate, ask_modulated_signal)
plt.title('ASK Modulated Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(np.arange(len(fsk_modulated_signal)) / sampling_rate, fsk_modulated_signal)
plt.title('FSK Modulated Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()

print("Original Binary Data:", binary_data)
