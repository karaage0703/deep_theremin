import serial
from time import sleep
from multiprocessing import Value, Process
import psonic


def sonic_pi(freq):
    psonic.use_synth(psonic.PULSE)
    while True:
        with psonic.Fx(psonic.REVERB):
            print(freq.value)
            psonic.play(freq.value, release=0.2)
            sleep(0.08)


def data_communication(freq):
    print('Waiting for connections...')
    while True:
        recv = ser.readline()
        try:
            freq.value = float(recv)
            print('set freq:' + str(freq.value))

        except:
            print('invalid message')
            

if __name__ == '__main__':
    ser = serial.Serial('/dev/serial0', 115200, timeout = 0.1)

    freq = Value('d', 70.0)

    process1 = Process(target=sonic_pi, args=[freq])
    process2 = Process(target=data_communication, args=[freq])

    process1.start()
    process2.start()

