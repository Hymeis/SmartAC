For client side(raspberry Pi):

run "libcamera-vid -n -t 0 --width 1920 --height 1080 --framerate 30 --inline --listen -o tcp://172.20.10.2:8888" on one terminal

run "python3 thermal.py 172.20.10.13:8890" on the other terminal

run "python3 fan.py" on the other terminal


For server side:

run "python3 detect.py --source=tcp://172.20.10.13:8888 --classes 0"

change the ip address based on the actual ip address for client and server