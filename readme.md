## Smart AC ##
This project involves designing an IoT system in Python to dynamically adjust multiple fan speeds based on detected human presence and respective temperatures

## Major Components needed ##
<ul>
  <li> 5th Gen Raspberry Pi </li>
  <li> NKX90640 Thermal Camera </li>
  <li> Raspberry Pi Camera </li>
  <li> 4x Raspberry Pi Fans </li>
  <li> (Optional) BreadBoard </li>
</ul>

## For client side(raspberry Pi): ##

  Run "libcamera-vid -n -t 0 --width 1920 --height 1080 --framerate 30 --inline --listen -o tcp://172.20.10.2:8888" on 1st terminal

  Run "python3 thermal.py 172.20.10.13:8890" on 2nd terminal

  Run "python3 fan.py" on 3rd terminal


## For server side: ##

  Run "python3 detect.py --source=tcp://172.20.10.13:8888 --classes 0"

  Change the ip address based on the actual ip address for client and server
