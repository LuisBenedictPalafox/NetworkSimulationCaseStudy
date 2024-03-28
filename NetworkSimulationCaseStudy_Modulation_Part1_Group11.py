import numpy as np
import matplotlib.pyplot as plt

# Part 1 - Modulation Technique
# Set parameters
bit_duration = 1              # Duration of each bit in seconds
num_bits = 16                 # Number of bits in the binary data
sample_rate = 500             # Number of samples per second
binary_data = np.random.randint(0, 2, num_bits)  # Generate random 16-bit binary data

# ASK (Amplitude Shift Keying) parameters
ask_carrier_frequency = 50
ask_time = np.arange(0, num_bits * bit_duration, bit_duration / sample_rate)
ask_carrier = np.sqrt(2 / bit_duration) * np.sin(2 * np.pi * ask_carrier_frequency * ask_time)

# FSK (Frequency Shift Keying) parameters
fsk_carrier_frequency0 = 25  # Carrier frequency for bit 0
fsk_carrier_frequency1 = 50  # Carrier frequency for bit 1
fsk_time = np.arange(0, num_bits * bit_duration, bit_duration / sample_rate)
fsk_carrier0 = np.sqrt(2 / bit_duration) * np.sin(2 * np.pi * fsk_carrier_frequency0 * fsk_time)
fsk_carrier1 = np.sqrt(2 / bit_duration) * np.sin(2 * np.pi * fsk_carrier_frequency1 * fsk_time)

# Modulation
ask_modulated = np.zeros(len(ask_time))
fsk_modulated = np.zeros(len(fsk_time))
for i in range(num_bits):
    if binary_data[i] == 1:
        ask_modulated[i * sample_rate:(i + 1) * sample_rate] = ask_carrier[i * sample_rate:(i + 1) * sample_rate]
        fsk_modulated[i * sample_rate:(i + 1) * sample_rate] = fsk_carrier1[i * sample_rate:(i + 1) * sample_rate]
    else:
        fsk_modulated[i * sample_rate:(i + 1) * sample_rate] = fsk_carrier0[i * sample_rate:(i + 1) * sample_rate]

# Display seconds for ASK and FSK modulated signals
display_seconds_ask = 3
display_seconds_fsk = 3

# Plotting all in one figure
plt.figure(figsize=(12, 18))

# Binary data bits
plt.subplot(9, 1, 1)
plt.stem(binary_data, linefmt='b-', markerfmt='bo', basefmt=' ')
plt.title('Binary Data Bits')
plt.xlabel('Bit Index')
plt.ylabel('Bit Value')
plt.ylim(-0.5, 1.5)
plt.grid(True)

# Message signal in polar form
plt.subplot(9, 1, 2)
plt.plot(ask_time, np.where(binary_data.repeat(sample_rate) > 0, 1, -1), 'r')
plt.title('Message Signal in Polar Form')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# ASK carrier signal (display only 1 second for clarity)
plt.subplot(9, 1, 3)
plt.plot(ask_time[:sample_rate], ask_carrier[:sample_rate], 'g')
plt.title('ASK Carrier Signal (1 second)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# ASK modulated signal
plt.subplot(9, 1, 4)
plt.plot(ask_time, ask_modulated, 'm')
plt.title('ASK Modulated Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# ASK modulated signal for desired seconds
samples_to_display_ask = display_seconds_ask * sample_rate
plt.subplot(9, 1, 5)
plt.plot(ask_time[:samples_to_display_ask], ask_modulated[:samples_to_display_ask], 'm')
plt.title(f'ASK Modulated Signal (First {display_seconds_ask} seconds)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# FSK carrier signal for Logic 0 (display only 1 second for clarity)
plt.subplot(9, 1, 6)
plt.plot(fsk_time[:sample_rate], fsk_carrier0[:sample_rate], 'c')
plt.title('FSK Carrier Signal (Logic 0, 1 second)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# FSK carrier signal for Logic 1 (display only 1 second for clarity)
plt.subplot(9, 1, 7)
plt.plot(fsk_time[:sample_rate], fsk_carrier1[:sample_rate], 'y')
plt.title('FSK Carrier Signal (Logic 1, 1 second)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# FSK modulated signal
plt.subplot(9, 1, 8)
plt.plot(fsk_time, fsk_modulated, 'k')
plt.title('FSK Modulated Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# FSK modulated signal for desired seconds
samples_to_display_fsk = display_seconds_fsk * sample_rate
plt.subplot(9, 1, 9)
plt.plot(fsk_time[:samples_to_display_fsk], fsk_modulated[:samples_to_display_fsk], 'k')
plt.title(f'FSK Modulated Signal (First {display_seconds_fsk} seconds)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()


