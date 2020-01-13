# -*- coding:utf-8 -*-
import socket

host = "xx.xx.xx.xx" # ip address or host name
port = 8888

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((host, port)) 

def send(freq):
    freq = bytes(str(freq).encode('utf-8'))
    client.send(freq)

if __name__ == '__main__':
    send(100)
