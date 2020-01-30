import serial
from time import sleep
from multiprocessing import Value, Process
import psonic


def sonic_pi(freq):
    psonic.use_synth(psonic.PULSE)
    while True:
        with psonic.Fx(psonic.REVERB):
            if instrument.value == 0:
                psonic.use_synth(psonic.PULSE)
            if instrument.value == 1:
                psonic.use_synth(psonic.SAW)
            if instrument.value == 2:
                psonic.use_synth(psonic.FM)

            psonic.play(freq.value, release=0.2)
            sleep(0.08)



def data_communication(freq):
    print('Waiting for connections...')
    while True:
        recv = ser.readline()
        print(recv)

        try:
            recv = recv.decode()
            f, inst = recv.split(',')
            freq.value = float(f)
            if inst == 'choki':
                instrument.value = 0
            if inst == 'gu':
                instrument.value = 1
            if inst == 'pa':
                instrument.value = 2

            print('set freq:' + str(freq.value))
            print('instrument:' + str(instrument.value))

        except:
            print('invalid message')
            

if __name__ == '__main__':
    ser = serial.Serial('/dev/serial0', 115200, timeout = 0.2)

    freq = Value('d', 70.0)
    instrument = Value('i', 0)

    process1 = Process(target=sonic_pi, args=[freq])
    process2 = Process(target=data_communication, args=[freq])

    process1.start()
    process2.start()

