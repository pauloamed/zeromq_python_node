import zmq
import random
import sys
import time
import json


context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5555")

while True:
    socket.send(b'1 { "name":"John", "age":30, "city":"New York"}')
    time.sleep(1)
