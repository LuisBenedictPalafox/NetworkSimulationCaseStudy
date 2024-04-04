import numpy as np
import socket
import pickle
import matplotlib.pyplot as plt

# Define the receiver's address and port
SERVER_ADDRESS = 'localhost' #change to proper ip address when needed
SERVER_PORT = 3303

# Function to convert ASCII to binary
def ascii_to_binary(ascii_string):
    return ''.join(format(ord(char), '08b') for char in ascii_string)

# Function to modulate the binary data using ASK and FSK
def modulate_binary_data(binary_data, bit_duration, samples_per_second, ask_carrier_freq, fsk_carrier_freq0, fsk_carrier_freq1):
    num_bits = len(binary_data)
    ask_time = np.arange(0, num_bits * bit_duration, bit_duration / samples_per_second)
    fsk_time = np.arange(0, num_bits * bit_duration, bit_duration / samples_per_second)
    
    # ASK modulation
    ask_carrier_wave = np.sqrt(2 / bit_duration) * np.sin(2 * np.pi * ask_carrier_freq * ask_time)
    ask_modulated_signal = np.zeros(len(ask_time))
    for i in range(num_bits):
        if binary_data[i] == 1:
            ask_modulated_signal[i * samples_per_second:(i + 1) * samples_per_second] = ask_carrier_wave[i * samples_per_second:(i + 1) * samples_per_second]
    
    # FSK modulation
    fsk_carrier_wave0 = np.sqrt(2 / bit_duration) * np.sin(2 * np.pi * fsk_carrier_freq0 * fsk_time)
    fsk_carrier_wave1 = np.sqrt(2 / bit_duration) * np.sin(2 * np.pi * fsk_carrier_freq1 * fsk_time)
    fsk_modulated_signal = np.zeros(len(fsk_time))
    for i in range(num_bits):
        if binary_data[i] == 1:
            fsk_modulated_signal[i * samples_per_second:(i + 1) * samples_per_second] = fsk_carrier_wave1[i * samples_per_second:(i + 1) * samples_per_second]
        else:
            fsk_modulated_signal[i * samples_per_second:(i + 1) * samples_per_second] = fsk_carrier_wave0[i * samples_per_second:(i + 1) * samples_per_second]

    return binary_data, ask_modulated_signal, fsk_modulated_signal, samples_per_second, ask_time, fsk_time, ask_carrier_wave, fsk_carrier_wave0, fsk_carrier_wave1

def main():
    try:
        # Set parameters
        bit_duration = 1              # Duration of each bit in seconds
        samples_per_second = 500      # Number of samples per second
        ask_carrier_freq = 50         # ASK carrier frequency
        fsk_carrier_freq0 = 25        # FSK carrier frequency for bit 0
        fsk_carrier_freq1 = 50        # FSK carrier frequency for bit 1
        
        # Accept custom ASCII input and convert to binary
        ascii_input = input("Enter an ASCII string: ")
        binary_data_str = ascii_to_binary(ascii_input)
        binary_data = np.array([int(bit) for bit in binary_data_str])
        
        # Modulate the binary data
        binary_data, ask_modulated_signal, fsk_modulated_signal, samples_per_second, ask_time, fsk_time, ask_carrier_wave, fsk_carrier_wave0, fsk_carrier_wave1 = modulate_binary_data(
            binary_data,
            bit_duration,
            samples_per_second,
            ask_carrier_freq,
            fsk_carrier_freq0,
            fsk_carrier_freq1
        )
        
        # Create a dictionary to hold the data to be sent
        data_to_send = {
            'binary_data': binary_data,
            'ask_modulated_signal': ask_modulated_signal,
            'fsk_modulated_signal': fsk_modulated_signal,
            'samples_per_second': samples_per_second
        }
        
        # Serialize the data using pickle
        serialized_data = pickle.dumps(data_to_send)
        
        # Create a TCP/IP socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Connect to the server
            sock.connect((SERVER_ADDRESS, SERVER_PORT))
            
            # Send the data
            sock.sendall(serialized_data)
            print("Data sent successfully.")

            # Close the socket
            sock.close()
    
    except KeyboardInterrupt:
        print("Sender interrupted by user.")

    # Plotting for Part 1 (Modulation) after data is sent
    plt.figure(figsize=(12, 9))

    # Binary data bits
    plt.subplot(3, 1, 1)
    plt.stem(binary_data, linefmt='b-', markerfmt='bo', basefmt=' ')
    plt.title('Binary Data Bits')
    plt.xlabel('Bit Index')
    plt.ylabel('Bit Value')
    plt.ylim(-0.5, 1.5)
    plt.grid(True)

    # # Message signal in polar form
    # plt.subplot(6, 1, 2)
    # plt.plot(ask_time, np.where(binary_data.repeat(samples_per_second) > 0, 1, -1), 'r')
    # plt.title('Message Signal in Polar Form')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.grid(True)

    # ASK carrier signal (display only 1 second for clarity)
    plt.subplot(3, 1, 2)
    plt.plot(ask_time[:samples_per_second], ask_carrier_wave[:samples_per_second], 'g')
    plt.title('ASK Carrier Signal (1 second)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # ASK modulated signal
    plt.subplot(3, 1, 3)
    plt.plot(ask_time, ask_modulated_signal, 'm')
    plt.title('ASK Modulated Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.tight_layout()

    plt.figure(figsize=(12, 9))

    # FSK carrier signal for Logic 0 (display only 1 second for clarity)
    plt.subplot(3, 1, 1)
    plt.plot(fsk_time[:samples_per_second], fsk_carrier_wave0[:samples_per_second], 'c')
    plt.title('FSK Carrier Signal (Logic 0, 1 second)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # FSK carrier signal for Logic 1 (display only 1 second for clarity)
    plt.subplot(3, 1, 2)
    plt.plot(fsk_time[:samples_per_second], fsk_carrier_wave1[:samples_per_second], 'y')
    plt.title('FSK Carrier Signal (Logic 1, 1 second)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # FSK modulated signal
    plt.subplot(3, 1, 3)
    plt.plot(fsk_time, fsk_modulated_signal, 'k')
    plt.title('FSK Modulated Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.tight_layout()

    # To clearly see the sine wave forms in custom time range
    # Calculate the indices for the nth (start) to nth (end) second
    start_time = 6
    end_time = 9
    start_index = start_time * samples_per_second  # Example: Start at 6th second
    end_index = end_time * samples_per_second    # Example: End at 9th second

    # Plot the ASK modulated signal from the start_second to the end_second
    plt.figure(figsize=(12, 6))
    plt.plot(ask_time[start_index:end_index], ask_modulated_signal[start_index:end_index], 'm')
    plt.title('ASK Modulated Signal (from time ' + str(start_time) + ' to time ' + str(end_time) + ')')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.tight_layout()

    # Plot the FSK modulated signal from the start_second to the end_second
    plt.figure(figsize=(12, 6))
    plt.plot(fsk_time[start_index:end_index], fsk_modulated_signal[start_index:end_index], 'k')
    plt.title('FSK Modulated Signal (from time ' + str(start_time) + ' to time ' + str(end_time) + ')')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    main()
