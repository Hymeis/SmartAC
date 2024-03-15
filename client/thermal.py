import socket
import pickle  # Used for serializing Python objects
import sys
import time
import board
import busio
import numpy as np
import adafruit_mlx90640

def parse_host_port():
    if len(sys.argv) != 2:
        print("Usage: python3 client.py <host:port>")
        sys.exit(1)
    try:
        host, port = sys.argv[1].split(":")
        port = int(port)
        return host, port
    except ValueError:
        print("Invalid host:port format")
        sys.exit(1)

# Define host and port
HOST, PORT = parse_host_port()


# Setup I2C
i2c = busio.I2C(board.SCL, board.SDA, frequency=400000) # Setup I2C
mlx = adafruit_mlx90640.MLX90640(i2c) # Begin MLX90640 with I2C comm
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ # Set refresh rate
mlx_shape = (24, 32) # Shape of the MLX90640 sensor array


# Create a TCP/IP socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    server_socket.bind((HOST, PORT))  # Bind the socket to the address and port
    server_socket.listen()  # Listen for incoming connections
    print(f'Server is listening on {HOST}:{PORT}...')

    # Accept a connection
    client_socket, client_address = server_socket.accept()
    print(f'Connected to {client_address}')

    try:
        while True:
            # Read thermal values from MLX90640 sensor
            frame = np.zeros((24*32,))
            mlx.getFrame(frame)  # Read MLX temperatures into frame var
            data_array = np.reshape(frame, mlx_shape)  # Reshape to 24x32 array
            
            # Serialize data_array using pickle
            serialized_data_array = pickle.dumps(data_array)
            

            # Send the serialized data_array to the client
            client_socket.sendall(serialized_data_array)

            # Delay to match the MLX90640 refresh rate
            time.sleep(1 / mlx.refresh_rate)

    except KeyboardInterrupt:
        print("Server shutting down...")
        client_socket.close()
    finally:
        # Close the connection
        client_socket.close()
