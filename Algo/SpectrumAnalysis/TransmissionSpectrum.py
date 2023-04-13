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


# PARAMETERS
WAIT_TIME = 1
CH_A=0
CH_B = 1
CH_C = 2




class TransmissionSpectrum:
    def __init__(self,init_wavelength = 772,final_wavelength = 772.3,Python_Control = True):
        '''

        :param directory: directory to load traces (relevant when Python_Control=False)
        :param init_wavelength: initial wavelength for scan [nm]
        :param final_wavelength:
        :param Python_Control:
        :param decimation:
        '''
        from BasicInstrumentsControl.PicoControl.PicoControl_4channels import PicoControl as Pico
        from BasicInstrumentsControl.PicoControl.PicoControl_4channels import PicoScopeControl as Scope
        from BasicInstrumentsControl.PicoControl.PicoControl_4channels import PicoSigGenControl as SigGen
        from BasicInstrumentsControl.Laser.LaserControl import LaserControl as Laser


        if Python_Control:
            #connect to instruments
            self.Pico = Pico()
            self.SigGen = SigGen(pico=self.Pico,pk_to_pk_voltage = 0.8, offset_voltage = 0, frequency = 10,wave_type = 'TRIANGLE') # frequency = 10
            self.Scope = Scope(pico=self.Pico)
            self.Laser = Laser()

            # define variables
            self.final_wavelength = final_wavelength
            self.init_wavelength = init_wavelength
            self.single_scan_width = self.SigGen.calculate_scan_width()
            print('The scan width in nm is:', self.single_scan_width)
            self.Laser.tlb_set_wavelength(self.init_wavelength)
            self.detector_noise = 0




    def get_wide_spectrum(self,parmeters_by_console):
        self.total_spectrum = []
        self.partial_spectrum = []

        self.total_Cosy_spectrum = []
        self.partial_Cosy_spectrum = []

        #self.total_A_chnnel_spectrum = []
        #self.partial_A_chnnel_spectrum = []
        self.trace_limits = []
        init_limits = 0
        self.SigGen_spectrum = []

        if parmeters_by_console:
            # for delete the detector's noise
            try:
                self.detector_noise = 0
                # self.check_detector_noise()
            except Exception:
                raise

            print("Enter initial wavelength for scan in [nm]:")
            self.init_wavelength = float(input())

            print("Enter final wavelength for scan in [nm]:")
            self.final_wavelength = float(input())

        self.Laser.tlb_set_wavelength(self.init_wavelength)
        time.sleep(5*WAIT_TIME)

        # jump between wavelengths and take traces
        for i in np.arange(self.init_wavelength, self.final_wavelength, self.single_scan_width):
            self.Laser.tlb_set_wavelength(i)
            # added wait time to make sure the laser moved to it's new wavelength
            time.sleep(WAIT_TIME)
            # # first scan to calibrate the range
            # self.Scope.calibrate_range()
            # take trace from the scope
            self.partial_spectrum.append(self.Scope.get_trace()[CH_B])# + self.Scope.calibrate_trace_avg_voltage * 1000)

            # list of indexes where the scop trace ends
            self.trace_limits.append(init_limits + len(self.Scope.get_trace()[CH_B]))
            init_limits = init_limits + len(self.Scope.get_trace()[CH_B])
            self.SigGen_spectrum.append(self.Scope.get_trace()[CH_A])

            #self.partial_A_chnnel_spectrum.append(self.Scope.get_trace()[CH_A])

            # taking Cosy scan between [778, final wavlength] from channel C
            if i > 779.2:
                self.partial_Cosy_spectrum.append(self.Scope.get_trace()[CH_C])# + self.Scope.calibrate_trace_avg_voltage * 1000)

        # patch the traces 771 781
        self.total_spectrum = np.concatenate(self.partial_spectrum)
        self.total_spectrum = [float(item) for item in self.total_spectrum]
        self.total_spectrum = np.array(self.total_spectrum)

        # create Cosy scan if exist
        if self.partial_Cosy_spectrum != []:
            self.total_Cosy_spectrum = np.concatenate(self.partial_Cosy_spectrum)
            self.total_Cosy_spectrum = [float(item) for item in self.total_Cosy_spectrum]
            self.total_Cosy_spectrum = np.array(self.total_Cosy_spectrum)

        self.SigGen_spectrum = np.concatenate(self.SigGen_spectrum)
        self.SigGen_spectrum = np.array([float(item) for item in self.SigGen_spectrum])


        # create vector of wavelengths in the scan
        self.get_scan_wavelengths()

    def check_detector_noise(self):
        # precaution - run the function only if the laser turned on by the user
        if self.Laser.tlb_query('OUTPut:STATe?') != '1':
            raise TypeError('Error. Turn on the laser and try again')

        # turn the laser off and take mean of pico trace
        self.Laser.tlb_query('OUTPut:STATe 0')
        time.sleep(WAIT_TIME*3)
        if self.Laser.tlb_query('OUTPut:STATe?') == '0':
            print('The laser turned off to find the detector noise')
        self.detector_noise = np.mean(self.Scope.get_trace()[CH_B])

        # turn the laser on
        try:
            self.Laser.tlb_query('OUTPut:STATe 1')
        except Exception:
            raise TypeError('Error in finding the detectror\'s noise')

        time.sleep(WAIT_TIME*5)
        if self.Laser.tlb_query('OUTPut:STATe?') == '1':
            print('The laser turned on, the noise is '+str(np.round(self.detector_noise,3)))
        #if self.detector_noise == 0:
       #     print('Wow, error:(')
       # else:
        #    print('The detector\'s noise is '+str(np.round(self.detector_noise,3)))

    def get_scan_wavelengths(self):
        m_wavenumber_transmitted = (self.final_wavelength - self.init_wavelength + self.single_scan_width) / len(
            self.total_spectrum)
        self.scan_wavelengths = (m_wavenumber_transmitted * np.arange(0, len(self.total_spectrum)) + (
                    self.init_wavelength - self.single_scan_width / 2))
        # self.scan_wavelengths = self.scan_wavelengths[0:-1]

    def read_csv(self, filename):
        csv_data = pd.read_csv(filename, sep=',', header=None)
        return csv_data.values

    def filter_spectrum(self, filter_type = 'high',filter_order=4, filter_critical_freq=0.9):
        b, a = signal.butter(filter_order, filter_critical_freq, btype=filter_type)
        return signal.filtfilt(b, a, self.total_spectrum)

    def plot_transmission_spectrum(self, Waveguide, Cosy, decimation):
        plt.figure()
        plt.title('Transmission Spectrum')
        plt.xlabel('Wavelength[nm]')
        plt.ylabel('Voltage[mV]')
        #plt.legend('WG scan', 'Cosy scan')
        plt.grid(True)
        plt.plot(self.scan_wavelengths[0:-1:decimation], Waveguide[0:-1:decimation],'r')
        # plot cosy scan
        #start_index = len(self.scan_wavelengths[0:-1]) - len(Cosy[0:-1])
        #plt.plot(self.scan_wavelengths[start_index:-1:decimation], Cosy[0:-1:decimation],'g')
        #plt.show()

    def X_axis_resolution_check(self, Waveguide, decimation):
        '''
        plot the scan with the limits between scope traces.
        this function isn't necessary in the scan process, use only for checking that the scan precise in X axis
        For changing the resolution in axis X-
        a. change preTriggerSamples & postTriggerSamples in PicoControl_4channels
        b. change the Frequency of SigGen() here in line 48
        '''
        plt.figure()
        plt.title('Transmission Spectrum X axis check')
        plt.xlabel('Wavelength[nm]')
        plt.ylabel('Voltage[mV]')
        plt.grid(True)
        plt.plot(self.scan_wavelengths[0:-1:decimation], Waveguide[0:-1:decimation],'r')
        plt.plot(self.scan_wavelengths[0:-1:decimation], self.SigGen_spectrum[0:-1:decimation]/1000, 'r')
        # plt.show()
        self.trace_limits = [int(i) for i in self.trace_limits]
        for i in self.trace_limits:
            if i > 50 and i < (len(Waveguide)-50):
                print(i)
                plt.plot(self.scan_wavelengths[i-20: i+20], Waveguide[i-20: i+20], 'y')
        plt.show()

    def save_figure_and_data(self,dist_root,spectrum_data,Cosy,decimation,filename = 'Transmission_spectrum', mkdir = True, detector_noise_val=12, trace_limits = 0):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        if mkdir:
            # create directory
            self.transmission_directory_path = os.path.join(dist_root,timestr)
            os.mkdir(self.transmission_directory_path)

        # save figure
        plt.savefig(os.path.join(self.transmission_directory_path,timestr+filename+'.png'))

        # save data as csv
        np.savetxt(os.path.join(self.transmission_directory_path,timestr+filename+'.csv'), spectrum_data[0:-1:decimation], delimiter=',')
        np.savetxt(os.path.join(self.transmission_directory_path, timestr + filename + '_cosy_data.csv'), Cosy[0:-1:decimation], delimiter=',')

        # save python data
        np_filename = timestr + filename + '.npz'
        np_root = os.path.join(self.transmission_directory_path,np_filename)
        # detector's noise &
        detector_noise_val = np.ones(100)*detector_noise_val
        if trace_limits != 0:
            trace_limits = [int(t/decimation) for t in trace_limits]
        np.savez(np_root, spectrum = spectrum_data[0:-1:decimation], wavelengths = self.scan_wavelengths[0:-1:decimation],
                 cosy_spectrum = Cosy[0:-1:decimation], cosy_wavelengths = self.scan_wavelengths[len(self.scan_wavelengths[0:-1]) - len(Cosy[0:-1]):-1:decimation], detector_noise = detector_noise_val, trace_limits = trace_limits)
        return np_root

if __name__ == "__main__":
    try:
        Python_Control=False
        o = TransmissionSpectrum(init_wavelength = 772,final_wavelength = 781,Python_Control = True)
        # get spectrum
        o.get_wide_spectrum(parmeters_by_console=True)
        decimation = 10
        o.X_axis_resolution_check(o.total_spectrum, decimation=decimation)
        o.plot_transmission_spectrum(o.total_spectrum, o.total_Cosy_spectrum, decimation=decimation)
        o.save_figure_and_data(r'C:\Users\Lab2\qs-labs\R&D - Lab\Chip Tester\Spectrum_transmission\unnamed scans',o.total_spectrum,
                               o.total_Cosy_spectrum, decimation, 'Test', detector_noise_val = o.detector_noise, trace_limits = o.trace_limits)
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