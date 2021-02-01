import scipy.io as sio
from scipy.io.wavfile import read, write

import matplotlib.pyplot as plt

import numpy as np

import sounddevice as sd



Fs, data = read('../Music/genres_original/blues/blues.00000.wav')


plt.figure()
plt.plot(data)
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Waveform test audio')
plt.show()