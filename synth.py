import math
import numpy as np
import pyaudio
import struct
from time import sleep

def play(stream,data):
     chunk = 1024
     sp = 0
     buffer = data[sp:sp+chunk]
     while buffer:
         stream.write(buffer)
         sp = sp + chunk
         buffer = data[sp:sp+chunk]

def createData(freqList = [440], start_pos=0, amplifer=1.0):
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

if __name__ == '__main__':
     p = pyaudio.PyAudio()
     stream = p.open(format=pyaudio.paInt16,channels=1, rate=44100, output=1)

     pos = 0
     freq = 440
     count = 0
     count_max = 100
     data, pos = createData(freqList = [freq], start_pos=pos, amplifer=0.5)
     while True :
         with open('freq.txt') as f:
             try:
                 freq = float(f.read())
             except:
                 pass
         data, pos = createData(freqList = [freq], start_pos=pos, amplifer=0.5)
         play(stream,data)

     stream.close()
     p.terminate()
