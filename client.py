# -*- coding:utf-8 -*-
import socket

host = "localhost" # ip address or host name
port = 8888 # same as host

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((host, port))

def send(freq):
    global client
    freq = bytes(str(freq).encode('utf-8'))
    try:
        client.send(freq)
    except:
        print('can not send')


if __name__ == '__main__':
    send(100)
