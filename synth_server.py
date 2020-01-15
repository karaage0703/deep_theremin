import math
import numpy as np
import pyaudio
import struct
from time import sleep
import socket
from multiprocessing import Value, Process
# import psonic

host = "localhost" # Input ip address or host name
port = 8888 # same as client

serversock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serversock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
serversock.bind((host,port))
serversock.listen(10)


def play(stream,data):
    chunk = 1024
    sp = 0
    buffer = data[sp:sp+chunk]
    while buffer:
        stream.write(buffer)
        sp = sp + chunk
        buffer = data[sp:sp+chunk]


def create_data(freqList = [440], start_pos=0, amplifer=1.0):
    data = []
    amp = amplifer / len(freqList)

    end_pos = start_pos + 0.05 * 44100
    for n in np.arange(start_pos, end_pos):
        s = 0.0
        for f in freqList:
            s += amp * np.sin(2 * np.pi * f * n / 44100)
        if s > 1.0:  s = 1.0
        if s < -1.0: s = -1.0
        data.append(s)
    data = [int(x * 32767.0) for x in data]

    data = struct.pack("h" * len(data), *data)

    return data, end_pos


def sound_out(freq):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,channels=1, rate=44100, output=1)

    pos = 0
    count = 0
    count_max = 100
    data, pos = create_data(freqList = [freq.value], start_pos=pos, amplifer=0.5)

    while True:
        print(freq.value)
        data, pos = create_data(freqList = [freq.value], start_pos=pos, amplifer=0.5)
        play(stream, data)

    stream.close()
    p.terminate()

def sonic_pi(freq):
    import psonic
    psonic.use_synth(psonic.PULSE)
    while True:
        with psonic.Fx(psonic.REVERB):
            print(freq.value)
            psonic.play(freq.value, release=0.2)
            sleep(0.08)


def data_communication(freq):
    print('Waiting for connections...')
    try:
        clientsock, client_address = serversock.accept() #接続されればデータを格納

        while True:
            recv = clientsock.recv(1024)
            freq.value = float(recv.decode())
            # print(freq)

    except KeyboardInterrupt:
        print('Interrupt!')
        serversock.close()


if __name__ == '__main__':

    freq = Value('d', 70)

    process1 = Process(target=sonic_pi, args=[freq])
    # process1 = Process(target=sound_out, args=[freq])
    process2 = Process(target=data_communication, args=[freq])

    process1.start()
    process2.start()

