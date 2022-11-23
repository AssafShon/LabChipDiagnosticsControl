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
import os


from BasicInstrumentsControl.PicoControl.PicoControl import PicoControl as Pico
from BasicInstrumentsControl.PicoControl.PicoControl import PicoScopeControl as Scope
from BasicInstrumentsControl.PicoControl.PicoControl import PicoSigGenControl as SigGen
from BasicInstrumentsControl.Laser.LaserControl import LaserControl as Laser

# PARAMETERS
WAIT_TIME = 1
CH_A=0
CH_B = 1



class TransmissionSpectrum:
    def __init__(self,directory='20220824-0002',init_wavelength = 772,final_wavelength = 781,Python_Control = True, parmeters_by_console = True):
        if Python_Control:
            self.Pico = Pico()
            self.SigGen = SigGen(pico=self.Pico,pk_to_pk_voltage = 0.8, offset_voltage = 0, frequency = 10,wave_type = 'TRIANGLE')
            self.Scope = Scope(pico=self.Pico)
            self.Laser = Laser()

            if parmeters_by_console:
                print("Enter initial wavelength for scan:")
                init_wavelength = float(input())
                print("Enter final wavelength for scan:")
                final_wavelength = float(input())

            self.final_wavelength = final_wavelength
            self.init_wavelength = init_wavelength
            self.single_scan_width = self.SigGen.calculate_scan_width()
            print('The scan width in nm is:',self.single_scan_width)
            self.Laser.tlb_set_wavelength(self.init_wavelength)
            time.sleep(5*WAIT_TIME)

            self.partial_spectrum = []
            # jump between wavelengths and take traces
            for i in np.arange(self.init_wavelength, self.final_wavelength, self.single_scan_width):
                self.Laser.tlb_set_wavelength(i)
                # added wait time to make sure the laser moved to it's new wavelength
                time.sleep(WAIT_TIME)
                # # first scan to calibrate the range
                self.Scope.calibrate_range()
                # take trace from the scope
                self.partial_spectrum.append(self.Scope.get_trace()[CH_B] + self.Scope.calibrate_trace_avg_voltage*1000)
                # patch the traces
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

    def filter_spectrum(self, filter_type = 'high',filter_order=4, filter_critical_freq=0.9):
        b, a = signal.butter(filter_order, filter_critical_freq, btype=filter_type)
        return signal.filtfilt(b, a, self.total_spectrum)

    def plot_spectrum(self, Y):
        m_wavenumber_transmitted = (self.final_wavelength-self.init_wavelength+self.single_scan_width)/len(self.total_spectrum)
        self.scan_wavelengths = m_wavenumber_transmitted*np.arange(0,len(self.total_spectrum))+(self.init_wavelength-self.single_scan_width/2)

        plt.figure()
        plt.title('Tansmission Spectrum')
        plt.xlabel('Wavelength[nm]')
        plt.ylabel('Voltage[mV]')
        plt.grid(True)
        plt.plot(self.scan_wavelengths, Y,'r')
        plt.show()

    def save_figure_and_data(self,dist_root,spectrum_data,decimation,filename = 'Transmission_spectrum'):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        # create directory
        directory_path = os.path.join(dist_root,timestr)
        os.mkdir(directory_path )
        # save figure
        plt.savefig(os.path.join(directory_path,timestr+filename+'.png'))
        #save data as csv
        np.savetxt(os.path.join(directory_path,timestr+filename+'.csv'), spectrum_data[0:-1:decimation], delimiter=',')
        #save python data
        np.savez(os.path.join(directory_path,timestr+filename+'.npz'), spectrum = spectrum_data[0:-1:decimation],wavelengths = self.scan_wavelengths[0:-1:decimation])

if __name__ == "__main__":
    try:
        o=TransmissionSpectrum(init_wavelength = 772,final_wavelength = 781,Python_Control = True)
        # o.plot_spectrum(o.total_spectrum)
        o.save_figure_and_data(r'C:\Users\Lab2\qs-labs\R&D - Lab\Chip Tester\Spectrum_transmission',[1,2,3],1000, 'Test')
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