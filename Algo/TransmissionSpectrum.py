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



import BasicInstrumentsControl.Laser.LaserControl as Laser
from BasicInstrumentsControl.PicoControl.PicoControl import PicoScopeControl as Scope
from BasicInstrumentsControl.PicoControl.PicoControl import PicoSigGenControl as SigGen

# PARAMETERS
WAIT_TIME = 2


class TransmissionSpectrum:
    def __init__(self,directory='20220824-0002',init_wavelength = 776e-9,final_wavelength = 781e-9,Python_Control = True):
        """"""
        if Python_Control:
            self.SigGen = SigGen(pk_to_pk_voltage = 2, offset_voltage = 0.0, frequency = 10,wave_type = 'RAMP_UP')
            self.single_scan_width = self.SigGen.calculate_scan_width()
            for i in np.arange(init_wavelength, final_wavelength, self.single_scan_width):
                Laser.tlb_set_wavelength(i)
                time.sleep(WAIT_TIME)
                self.spectrum_sheet.append(Scope.get_trace[0])
        else:
            self.directory = directory
            self.spectrum_parts = []
            self.total_spectrum = []
            for i in range(1, 1500, 50):
                filename = '20220824-0001_' + '{0:04}'.format(i) + '.csv'
                full_path = self.directory+'//'+ filename
                self.spectrum_parts.append(self.read_csv(full_path)[2:])

        self.total_spectrum = np.concatenate(self.spectrum_parts)
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

    def plot_spectrum_sheets(self,Y,num_of_plots):
        # Create Figure and Axes instances
        if num_of_plots>1:
            fig = plt.figure()
            ax1 = fig.add_axes(
                               xticklabels=[], ylim=(-10, 10))
            ax2 = fig.add_axes(
                               ylim=(-10, 10))
            ax1.plot(Y[0])
            ax2.plot(Y[1])
        else:
            # Make your plot, set your axes labels
            ax.plot(Y)
        plt.show()

    def plot_unfied_spectrum(self):
        wave_number_init = 776.5
        wave_number_final = 781
        m_wavenumber_transmitted = (wave_number_final-wave_number_init)/len(self.total_spectrum)
        wavenumber_transmitted = m_wavenumber_transmitted*np.arange(0,len(self.total_spectrum))+wave_number_init
        plt.plot(wavenumber_transmitted,self.total_spectrum)
        plt.show()

    def save_figure(self,dist_name):
        plt.savefig(dist_name)

if __name__ == "__main__":
    o=TransmissionSpectrum(init_wavelength = 776e-9,final_wavelength = 781e-9,Python_Control = True)
    # scan_spectrum = []
    # for i in range (2,6):
    #     scan_spectrum += [o.piezo_scan_spectrum(i)]
    # o.plot_spectrum(scan_spectrum,2)
    # o.plot_unfied_spectrum()
    # o.save_figure('fig_'+str(i))