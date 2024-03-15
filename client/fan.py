import RPi.GPIO as GPIO
import time
import subprocess
import socket
import json
import threading


fan_speed = [0, 0, 0, 0]


def start_server(host='172.20.10.13', port=9500):
    global fan_speed
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"Server listening on {host}:{port}")

        # Accept a new connection
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            try:
                while True:
                    # Receive data from the client
                    data = conn.recv(1024)
                    if not data:
                        # No more data from client, break the loop
                        break
                    # Deserialize the data to Python object
                    fan_speed = json.loads(data.decode())
                    print("Received fan_speed:", fan_speed)
            except ConnectionResetError:
                print("Connection reset by the client.")
            except Exception as e:
                print(f"An error occurred: {e}")
            finally:
                print("Connection closed.")


def start_fan():
    GPIO.setmode(GPIO.BCM)
    ##Set to false, other processes occupying the pin will be ignored
    GPIO.setwarnings(False)
    GPIO.setup(12, GPIO.OUT)
    GPIO.setup(13, GPIO.OUT)
    GPIO.setup(14, GPIO.OUT)
    GPIO.setup(15, GPIO.OUT)
    pwm = GPIO.PWM(12, 100)
    pwm1 = GPIO.PWM(13, 100)
    pwm2 = GPIO.PWM(14, 100)
    pwm3 = GPIO.PWM(15, 100)
    print("\nPress Ctrl+C to quit \n")
    dc = 0
    pwm.start(dc)
    pwm1.start(dc)
    pwm2.start(dc)
    pwm3.start(dc)
    try:
        while True:
            print("start running at the speed", fan_speed)
            dc, dc1, dc2, dc3 = fan_speed

            pwm.ChangeDutyCycle(dc)
            pwm1.ChangeDutyCycle(dc1)
            pwm2.ChangeDutyCycle(dc2)
            pwm3.ChangeDutyCycle(dc3)
            time.sleep(0.05)
    except KeyboardInterrupt:
        dc = 0
        pwm.ChangeDutyCycle(dc)
        pwm1.ChangeDutyCycle(dc)
        pwm2.ChangeDutyCycle(dc)
        pwm3.ChangeDutyCycle(dc)
        print("Ctrl + C pressed -- Ending program")


thread_server = threading.Thread(target=start_server, args=('172.20.10.13', 9500))
thread_fan = threading.Thread(target=start_fan)
thread_server.start()
thread_fan.start()
thread_server.join()
thread_fan.join()

