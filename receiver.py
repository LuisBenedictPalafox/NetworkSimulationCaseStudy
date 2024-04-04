import socket
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Define the receiver's address and port
SERVER_ADDRESS = 'localhost' #change to proper ip address when needed
SERVER_PORT = 3303

def binary_to_ascii(binary_string):
    return ''.join(chr(int(binary_string[i:i+8], 2)) for i in range(0, len(binary_string), 8))

def demodulate_data(data):
    binary_data = data['binary_data']
    ask_modulated_signal = data['ask_modulated_signal']
    fsk_modulated_signal = data['fsk_modulated_signal']
    samples_per_second = data['samples_per_second']
    num_bits = len(binary_data)

    # Generate carrier waves for demodulation
    bit_duration = 1
    ask_carrier_freq = 50
    fsk_carrier_freq0 = 25
    fsk_carrier_freq1 = 50
    ask_time = np.arange(0, num_bits * bit_duration, bit_duration / samples_per_second)
    fsk_time = np.arange(0, num_bits * bit_duration, bit_duration / samples_per_second)
    ask_carrier_wave = np.sqrt(2 / bit_duration) * np.sin(2 * np.pi * ask_carrier_freq * ask_time)
    fsk_carrier_wave0 = np.sqrt(2 / bit_duration) * np.sin(2 * np.pi * fsk_carrier_freq0 * fsk_time)
    fsk_carrier_wave1 = np.sqrt(2 / bit_duration) * np.sin(2 * np.pi * fsk_carrier_freq1 * fsk_time)

    # Demodulation
    ask_demodulated_data = np.zeros(num_bits)
    ask_correlation_values = np.zeros(num_bits)
    fsk_demodulated_data = np.zeros(num_bits)
    fsk_correlation_diffs = np.zeros(num_bits)

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

    return ask_demodulated_data, fsk_demodulated_data, ask_correlation_values, fsk_correlation_diffs

def plot_demodulated_signals(ask_demodulated_data, fsk_demodulated_data, ask_correlation_values, fsk_correlation_diffs):
    num_bits = len(ask_demodulated_data)

    # Plotting for Part 2 (Demodulation)
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

def main():
    try:
        # Create a TCP/IP socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Bind the socket to the server address and port
            sock.bind((SERVER_ADDRESS, SERVER_PORT))

            # Listen for incoming connections
            sock.listen(1)
            print('Waiting for a connection...')

            # Accept a connection
            connection, client_address = sock.accept()
            print('Connection from:', client_address)

            with connection:
                # Receive the data in chunks
                data_chunks = []
                while True:
                    chunk = connection.recv(4096)
                    if not chunk:
                        break
                    data_chunks.append(chunk)

                # Combine the chunks and deserialize the data
                serialized_data = b''.join(data_chunks)
                received_data = pickle.loads(serialized_data)

                # Perform demodulation
                ask_demodulated_data, fsk_demodulated_data, ask_correlation_values, fsk_correlation_diffs = demodulate_data(received_data)

                # Convert demodulated binary data back to ASCII
                ask_demodulated_ascii = binary_to_ascii(''.join(map(str, ask_demodulated_data.astype(int))))
                fsk_demodulated_ascii = binary_to_ascii(''.join(map(str, fsk_demodulated_data.astype(int))))

                # Print demodulated binary data as binary strings and as ASCII
                ask_demodulated_binary_str = ''.join(map(str, ask_demodulated_data.astype(int)))
                fsk_demodulated_binary_str = ''.join(map(str, fsk_demodulated_data.astype(int)))
                print("Demodulated binary data from ASK signal:", ask_demodulated_binary_str)
                print("Demodulated ASCII from ASK signal:", ask_demodulated_ascii)
                print("Demodulated binary data from FSK signal:", fsk_demodulated_binary_str)
                print("Demodulated ASCII from FSK signal:", fsk_demodulated_ascii)

                # Plot the demodulated signals (Part 2)
                plot_demodulated_signals(ask_demodulated_data, fsk_demodulated_data, ask_correlation_values, fsk_correlation_diffs)

                # Send an acknowledgment back to the sender
                connection.sendall(b'Acknowledgment: Data received')
    
    except KeyboardInterrupt:
        print("Receiver interrupted by user.")

if __name__ == '__main__':
    main()
