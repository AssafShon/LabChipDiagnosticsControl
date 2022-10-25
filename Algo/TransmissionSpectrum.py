'''
created at 14/08/22

@author: Assaf S.
'''

# This Class plots a transmition spectrum for the chip
# Tasks:
# 1. read each i'th (i=5) csv file to an array
# 2. attach arrays
# 3. plot

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy import signal
from datetime import datetime


from BasicInstrumentsControl.PicoControl.PicoControl import PicoControl as Pico
from BasicInstrumentsControl.PicoControl.PicoControl import PicoScopeControl as Scope
from BasicInstrumentsControl.PicoControl.PicoControl import PicoSigGenControl as SigGen
from BasicInstrumentsControl.Laser.LaserControl import LaserControl as Laser

# PARAMETERS
WAIT_TIME = 1


class TransmissionSpectrum:
    def __init__(self,directory='20220824-0002',init_wavelength = 776,final_wavelength = 781,Python_Control = True):
        if Python_Control:
            self.Pico = Pico()
            self.SigGen = SigGen(pico=self.Pico,pk_to_pk_voltage = 0.8, offset_voltage = 0, frequency = 10,wave_type = 'TRIANGLE')
            self.Scope = Scope(pico=self.Pico)
            self.Laser = Laser()

            self.final_wavelength = final_wavelength
            self.init_wavelength = init_wavelength
            self.single_scan_width = self.SigGen.calculate_scan_width()
            self.Laser.tlb_set_wavelength(self.init_wavelength)
            time.sleep(5*WAIT_TIME)

            self.partial_spectrum = []
            for i in np.arange(self.init_wavelength, self.final_wavelength, self.single_scan_width):
                self.Laser.tlb_set_wavelength(i)
                time.sleep(WAIT_TIME)

                self.partial_spectrum.append(self.Scope.get_trace()[1])
            self.total_spectrum = np.concatenate(self.partial_spectrum)
        else:
            self.directory = directory
            self.partial_spectrum = []
            self.total_spectrum = []
            for i in range(1, 1500, 50):
                filename = '20220824-0001_' + '{0:04}'.format(i) + '.csv'
                full_path = self.directory+'//'+ filename
                self.partial_spectrum.append(self.read_csv(full_path)[2:])

            self.total_spectrum = np.concatenate(self.partial_spectrum)
            self.total_spectrum = [item[1] for item in self.total_spectrum]

        self.total_spectrum = [float(item) for item in self.total_spectrum]
        self.total_spectrum = np.array(self.total_spectrum)

    def remove_infs(self):
        for i in range(1, len(self.total_spectrum)):
            if self.total_spectrum[i] == '-∞':
                self.total_spectrum[i] = '-10'

    def piezo_scan_spectrum(self, i):
        filename = '20220824-0001_' + '{0:04}'.format(i) + '.csv'
        full_path = self.directory + '//' + filename
        self.piezo_scan = self.read_csv(full_path)[2:]
        self.piezo_scan = [item[1] for item in self.piezo_scan]
        self.piezo_scan = [float(item) for item in self.piezo_scan]
        return self.piezo_scan

    def read_csv(self, filename):
        csv_data = pd.read_csv(filename, sep=',', header=None)
        return csv_data.values

    # def plot_spectrum_sheets(self,Y,num_of_plots):
    #     # Create Figure and Axes instances
    #     if num_of_plots>1:
    #         fig = plt.figure()
    #         ax1 = fig.add_axes(
    #                            xticklabels=[], ylim=(-10, 10))
    #         ax2 = fig.add_axes(
    #                            ylim=(-10, 10))
    #         ax1.plot(Y[0])
    #         ax2.plot(Y[1])
    #     else:
    #         # Make your plot, set your axes labels
    #         ax.plot(Y)
    #     plt.show()

    def filter_spectrum(self, filter_type = 'high',filter_order=4, filter_critical_freq=0.9):
        b, a = signal.butter(filter_order, filter_critical_freq, btype=filter_type)
        return signal.filtfilt(b, a, self.total_spectrum)
    def plot_spectrum(self, Y):
        m_wavenumber_transmitted = (self.final_wavelength-self.init_wavelength+self.single_scan_width)/len(self.total_spectrum)
        wavenumber_transmitted = m_wavenumber_transmitted*np.arange(0,len(self.total_spectrum))+(self.init_wavelength-self.single_scan_width/2)


        plt.figure()
        plt.title('Tansmission Spectrum')
        plt.xlabel('Wavelength[nm]')
        plt.ylabel('Voltage[mV]')
        plt.grid(True)
        plt.plot(wavenumber_transmitted, Y,'r')
        plt.show()

    def save_figure(self,dist_name):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        plt.savefig(timestr+'_'+dist_name)

if __name__ == "__main__":
    try:
        o=TransmissionSpectrum(init_wavelength = 774,final_wavelength = 781,Python_Control = True)
        o.plot_spectrum(o.total_spectrum)
        o.save_figure('transmission_spectrum.png')
        o.Pico.__del__()
        o.Laser.__del__()
    except:
        o.Pico.__del__()
        o.Laser.__del__()
        raise

    # scan_spectrum = []
    # for i in range (2,6):
    #     scan_spectrum += [o.piezo_scan_spectrum(i)]
    # o.plot_spectrum(scan_spectrum,2)
    # o.plot_unfied_spectrum()
    # o.save_figure('fig_'+str(i))