import numpy as np
import matplotlib.pyplot as plt

# Part 1 - Modulation Technique
# Set parameters
bit_duration = 1              # Duration of each bit in seconds
num_bits = 16               # Number of bits in the binary data
samples_per_second = 500      # Number of samples per second

# Accept custom 16-bit binary data
binary_data_str = input("Enter a 16-bit binary string: ")
binary_data = np.array([int(bit) for bit in binary_data_str])

# ASK (Amplitude Shift Keying) parameters
ask_carrier_freq = 50
ask_time = np.arange(0, num_bits * bit_duration, bit_duration / samples_per_second)
ask_carrier_wave = np.sqrt(2 / bit_duration) * np.sin(2 * np.pi * ask_carrier_freq * ask_time)

# FSK (Frequency Shift Keying) parameters
fsk_carrier_freq0 = 25  # Carrier frequency for bit 0
fsk_carrier_freq1 = 50  # Carrier frequency for bit 1
fsk_time = np.arange(0, num_bits * bit_duration, bit_duration / samples_per_second)
fsk_carrier_wave0 = np.sqrt(2 / bit_duration) * np.sin(2 * np.pi * fsk_carrier_freq0 * fsk_time)
fsk_carrier_wave1 = np.sqrt(2 / bit_duration) * np.sin(2 * np.pi * fsk_carrier_freq1 * fsk_time)

# Modulation
ask_modulated_signal = np.zeros(len(ask_time))
fsk_modulated_signal = np.zeros(len(fsk_time))
for i in range(num_bits):
    if binary_data[i] == 1:
        ask_modulated_signal[i * samples_per_second:(i + 1) * samples_per_second] = ask_carrier_wave[i * samples_per_second:(i + 1) * samples_per_second]
        fsk_modulated_signal[i * samples_per_second:(i + 1) * samples_per_second] = fsk_carrier_wave1[i * samples_per_second:(i + 1) * samples_per_second]
    else:
        fsk_modulated_signal[i * samples_per_second:(i + 1) * samples_per_second] = fsk_carrier_wave0[i * samples_per_second:(i + 1) * samples_per_second]

# Plotting for Part 1
plt.figure(figsize=(12, 18))

# Binary data bits
plt.subplot(7, 1, 1)
plt.stem(binary_data, linefmt='b-', markerfmt='bo', basefmt=' ')
plt.title('Binary Data Bits')
plt.xlabel('Bit Index')
plt.ylabel('Bit Value')
plt.ylim(-0.5, 1.5)
plt.grid(True)

# Message signal in polar form
plt.subplot(7, 1, 2)
plt.plot(ask_time, np.where(binary_data.repeat(samples_per_second) > 0, 1, -1), 'r')
plt.title('Message Signal in Polar Form')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# ASK carrier signal (display only 1 second for clarity)
plt.subplot(7, 1, 3)
plt.plot(ask_time[:samples_per_second], ask_carrier_wave[:samples_per_second], 'g')
plt.title('ASK Carrier Signal (1 second)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# ASK modulated signal
plt.subplot(7, 1, 4)
plt.plot(ask_time, ask_modulated_signal, 'm')
plt.title('ASK Modulated Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# FSK carrier signal for Logic 0 (display only 1 second for clarity)
plt.subplot(7, 1, 5)
plt.plot(fsk_time[:samples_per_second], fsk_carrier_wave0[:samples_per_second], 'c')
plt.title('FSK Carrier Signal (Logic 0, 1 second)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# FSK carrier signal for Logic 1 (display only 1 second for clarity)
plt.subplot(7, 1, 6)
plt.plot(fsk_time[:samples_per_second], fsk_carrier_wave1[:samples_per_second], 'y')
plt.title('FSK Carrier Signal (Logic 1, 1 second)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# FSK modulated signal
plt.subplot(7, 1, 7)
plt.plot(fsk_time, fsk_modulated_signal, 'k')
plt.title('FSK Modulated Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()

# Part 2 - Demodulation Technique
# Demodulation
ask_demodulated_data = np.zeros(num_bits)
ask_correlation_values = np.zeros(num_bits)  # For visualization
fsk_demodulated_data = np.zeros(num_bits)
fsk_correlation_diffs = np.zeros(num_bits)  # For visualization

# ASK Demodulation
for i in range(num_bits):
    sample_start = i * samples_per_second
    sample_end = (i + 1) * samples_per_second
    ask_correlation_values[i] = np.mean(ask_modulated_signal[sample_start:sample_end] * ask_carrier_wave[sample_start:sample_end])
    ask_demodulated_data[i] = int(ask_correlation_values[i] > 0.5)

# FSK Demodulation
for i in range(num_bits):
    sample_start = i * samples_per_second
    sample_end = (i + 1) * samples_per_second
    avg_50Hz = np.mean(fsk_modulated_signal[sample_start:sample_end] * fsk_carrier_wave1[sample_start:sample_end])
    avg_25Hz = np.mean(fsk_modulated_signal[sample_start:sample_end] * fsk_carrier_wave0[sample_start:sample_end])
    fsk_correlation_diffs[i] = avg_50Hz - avg_25Hz
    fsk_demodulated_data[i] = int(fsk_correlation_diffs[i] > 0)
# print(avg_25Hz)
# print(avg_50Hz)
# print(fsk_correlation_diffs)

# Plotting for Part 2
plt.figure(figsize=(14, 10))

# ASK Correlation Values
plt.subplot(4, 1, 1)
plt.plot(np.arange(num_bits), ask_correlation_values, 'g-o')
plt.title('ASK Correlation Values for Each Bit')
plt.xlabel('Bit Index')
plt.ylabel('Correlation Value')
plt.grid(True)

# ASK Demodulated Data
plt.subplot(4, 1, 2)
plt.stem(np.arange(num_bits), ask_demodulated_data, linefmt='b-', markerfmt='bo', basefmt=' ')
plt.title('ASK Demodulated Binary Data')
plt.xlabel('Bit Index')
plt.ylabel('Bit Value')
plt.ylim(-0.5, 1.5)
plt.grid(True)

# FSK Correlation Differences
plt.subplot(4, 1, 3)
plt.plot(np.arange(num_bits), fsk_correlation_diffs, 'r-o')
plt.title('FSK Correlation Differences for Each Bit (50Hz - 25Hz)')
plt.xlabel('Bit Index')
plt.ylabel('Correlation Difference')
plt.grid(True)

# FSK Demodulated Data
plt.subplot(4, 1, 4)
plt.stem(np.arange(num_bits), fsk_demodulated_data, linefmt='b-', markerfmt='bo', basefmt=' ')
plt.title('FSK Demodulated Binary Data')
plt.xlabel('Bit Index')
plt.ylabel('Bit Value')
plt.ylim(-0.5, 1.5)
plt.grid(True)

plt.tight_layout()
plt.show()
