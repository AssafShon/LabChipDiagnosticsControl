from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
from scipy import signal
import numpy as np


def lag_finder(y1, y2):
    n = len(y1)

    corr = signal.correlate(y2, y1, mode='same') / np.sqrt(signal.correlate(y1, y1, mode='same')[int(n/2)] * signal.correlate(y2, y2, mode='same')[int(n/2)])

    delay_arr = np.linspace(-0.5*n, 0.5*n, n)
    delay = delay_arr[np.argmax(corr)]
    print('y2 is ' + str(delay) + ' behind y1')

    plt.figure(1)
    plt.plot(delay_arr, corr)
    plt.title('Lag: ' + str(np.round(delay, 3)))
    plt.xlabel('Lag')
    plt.ylabel('Correlation coeff')
    plt.show()

delay_scan = 100
spectrum = np.load(r'20221116-170921Transmission_spectrum.npy')
# avg_dc = np.mean(spectrum)
spectrum_a = spectrum[int(len(spectrum)/2)-delay_scan:int(len(spectrum)/2)]
spectrum_b = spectrum[int(len(spectrum)/2):int(len(spectrum)/2)+delay_scan]
lag_finder(spectrum_a,spectrum_b)

plt.figure()
plt.plot(spectrum_a)
plt.plot(spectrum_b)

plt.figure()
plt.plot(spectrum)
# plt.plot(spectrum)

# # Sine sample with some noise and copy to y1 and y2 with a 1-second lag
#
# t = np.linspace(0, 1, 500, endpoint=False)
#
# y1 = signal.square(2 * np.pi  * t)
# sig2 = signal.square(2 * np.pi  * t+np.pi)
# y2 = sig2
#
# lag_finder(y1, y2)
#
# plt.figure(2)
# plt.plot(t,y1)
# plt.plot(t,y2)
#
# #
# t = np.linspace(0, 1, 500, endpoint=False)
# sig1 = signal.square(2 * np.pi  * t)
# sig2 = signal.square(2 * np.pi  * t+0.5)
# #
# corr = np.convolve(sig1, sig2)
# #
# plt.figure()
# plt.plot(corr)
# # plt.plot(t,sig1)
# # plt.plot(t,sig2)
# # plt.ylim(-2, 2)
# plt.grid()
# plt.show()
#
# # import matplotlib.pyplot as plt
# import numpy as np
#
# # First signal
# sig1 = np.sin()
#
# # Seconds signal with pi/4 phase shift. Half the size of sig1
# sig2 = np.sin(np.r_[-10:1:10] + np.pi/4)
#
# corr = np.correlate(a=sig1, v=sig2)
#
# plt.figure()
# plt.plot(corr)
# plt.plot(sig1)
# plt.plot(sig2)