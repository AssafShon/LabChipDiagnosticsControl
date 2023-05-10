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
CH_D = 3



class TransmissionSpectrum:
    def __init__(self,init_wavelength = 772,final_wavelength = 772.3,Python_Control = True):
        '''

        :param directory: directory to load traces (relevant when Python_Control=False)
        :param init_wavelength: initial wavelength for scan [nm]
        :param final_wavelength:
        :param Python_Control:
        :param decimation:
        '''
        from BasicInstrumentsControl.PicoControl.PicoControl_4channels_copy import PicoControl as Pico
        from BasicInstrumentsControl.PicoControl.PicoControl_4channels_copy import PicoScopeControl as Scope
        from BasicInstrumentsControl.PicoControl.PicoControl_4channels_copy import PicoSigGenControl as SigGen
        from BasicInstrumentsControl.Laser.LaserControl import LaserControl as Laser


        if Python_Control:
            #connect to instruments
            self.Pico = Pico()
            #self.SigGen = SigGen(pico=self.Pico,pk_to_pk_voltage=0.106, offset_voltage=0, frequency=10, wave_type='PS3000A_TRIANGLE')# frequency = 10
            self.SigGen = SigGen(pico=self.Pico, pk_to_pk_voltage=0.82, offset_voltage=0, frequency=10,
                                 wave_type='PS3000A_SQUARE')  # frequency = 10
            self.Scope = Scope(pico=self.Pico)
            self.Laser = Laser()

            # define variables
            self.final_wavelength = final_wavelength
            self.init_wavelength = init_wavelength
            self.single_scan_width = self.SigGen.calculate_scan_width()
            self.single_scan_width = 0.0224
            #self.single_scan_width = 1

            print('The scan width in nm is:', self.single_scan_width)
            self.Laser.tlb_set_wavelength(self.init_wavelength)
            self.detector_noise = 0

    def get_wide_spectrum(self,parmeters_by_console):
        self.total_spectrum = []
        self.partial_spectrum = []

        self.total_Cosy_spectrum = []
        self.partial_Cosy_spectrum = []
        self.trace_limits = []
        self.SigGen_spectrum = []
        init_limits = 0

        if parmeters_by_console:
            # for delete the detector's noise
            try:
                self.detector_noise = 0
                #self.get_detector_noise()
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

            # added wait time to make sure the laser moved to its new wavelength
            # time.sleep(WAIT_TIME)

            # add offset
            self.Scope.calibrate_range()
            # compensate of offset
            fixed_trace = [k + self.Scope.analog_offset for k in self.Scope.get_trace()[CH_B]]
            self.partial_spectrum.append(fixed_trace)

            # collect the signal generator signal (after the amplifier,before the laser)
            self.SigGen_spectrum.append(self.Scope.get_trace()[CH_A])

            #self.calibrate_range_test()
            self.SigGen_spectrum.append(self.Scope.calibrate_trace)

            # Optionally - collect list of indexes of scop trace limits
            self.trace_limits.append(init_limits + len(self.Scope.get_trace()[CH_B]))
            init_limits = init_limits + len(self.Scope.get_trace()[CH_B])

            # taking Cosy scan between [779.2, final wavelength] from channel C
            if i > 779.2:
                self.partial_Cosy_spectrum.append(self.Scope.get_trace()[CH_C])

        # patch the traces 771 781
        self.total_spectrum = np.concatenate(self.partial_spectrum)
        self.total_spectrum = np.array([float(item) for item in self.total_spectrum])

        # create Cosy scan if exist
        if self.partial_Cosy_spectrum != []:
            self.total_Cosy_spectrum = np.concatenate(self.partial_Cosy_spectrum)
            self.total_Cosy_spectrum = np.array([float(item) for item in self.total_Cosy_spectrum])

        self.SigGen_spectrum = np.concatenate(self.SigGen_spectrum)
        self.SigGen_spectrum = np.array([float(item) for item in self.SigGen_spectrum])


        # create vector of wavelengths in the scan
        self.get_scan_wavelengths()

    def get_wide_spectrum_DC_motor_only(self,parmeters_by_console):
        '''
        scan wavelength range with laser built in scan function.
        this way using only DC monitor without changing the piezo
        :param parmeters_by_console:
        :return:
        '''
        self.total_spectrum = []
        self.partial_spectrum = []

        self.total_Cosy_spectrum = []
        self.partial_Cosy_spectrum = []
        self.trace_limits = []
        self.SigGen_spectrum = []
        init_limits = 0

        if parmeters_by_console:
            # for delete the detector's noise
            try:
                self.detector_noise = 0
                #self.get_detector_noise()
            except Exception:
                raise

            print("Enter initial wavelength for scan in [nm]:")
            self.init_wavelength = float(input())

            print("Enter final wavelength for scan in [nm]:")
            self.final_wavelength = float(input())

        self.Laser.tlb_query('SOURce:WAVE:START {}'.format(self.init_wavelength))
        self.Laser.tlb_query('SOURce:WAVE:STOP {}'.format(self.final_wavelength))
        # need to create function in LaserControl

        self.Laser.tlb_set_wavelength(self.init_wavelength)
        time.sleep(5*WAIT_TIME)
        print('scanning...')
        self.Laser.tlb_query('OUTPut:SCAN:START')
        [self.SigGen_spectrum,self.total_spectrum,self.total_Cosy_spectrum,self.output_wavelength] = self.Scope.get_trace()

        time.sleep(5 * WAIT_TIME)
        print('Track mode: ')
        if self.Laser.tlb_query('OUTPut:TRACK?') == '0':
            print('Off')
        else:
            print('still On')

        # get_scan_wavelengths
        first_final_index = self.output_wavelength.index(np.max(self.output_wavelength))  # first max value
        reversed_output_wl = self.output_wavelength[first_final_index:first_final_index+499]
        reversed_output_wl.reverse()
        temp_index = reversed_output_wl.index(np.max(reversed_output_wl))
        last_final_index = first_final_index + len(reversed_output_wl) - temp_index - 1  # last max value
        self.final_wl_index = int((last_final_index + first_final_index)/2)

        sample_in_nm = (self.final_wavelength - self.init_wavelength) / self.final_wl_index
        self.scan_wavelengths = (sample_in_nm * np.arange(0, len(self.total_spectrum)))+self.init_wavelength
        self.output_wavelength = [j/100 for j in self.output_wavelength]

        #plot
        plt.figure()
        plt.title('Transmission Spectrum')
        plt.xlabel('Wavelength[nm]')
        plt.ylabel('Voltage[mV]')
        plt.legend(['full scan range', 'output wavelength', 'real range'])
        plt.grid(True)
        plt.plot(self.scan_wavelengths[0:-1], self.total_spectrum[0:-1], 'r')
        # plt.plot(self.scan_wavelengths[0:-1], self.SigGen_spectrum[0:-1], 'p')
        plt.plot(self.scan_wavelengths[0:-1], self.output_wavelength[0:-1], 'b')
        plt.plot(self.scan_wavelengths[0:self.final_wl_index], self.total_spectrum[0:self.final_wl_index], 'y')
        plt.show()

    def get_detector_noise(self):
        # precaution - run the function only if the laser turned on by the user
        if self.Laser.tlb_query('OUTPut:STATe?') != '1':
            raise TypeError('Error. Turn on the laser and try again')

        # turn the laser off and take mean of pico trace
        self.Laser.tlb_query('OUTPut:STATe 0')
        time.sleep(WAIT_TIME*4)
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
        # if self.detector_noise == 0:
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

    def filter_spectrum(self, filter_type='high', filter_order=4, filter_critical_freq=0.9):
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

    def save_figure_and_data(self,dist_root,spectrum_data,Cosy,decimation,filename = 'Transmission_spectrum', mkdir = True, detector_noise_val=12, trace_limits = []):
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

        # detector's noise, cosy & trace limits
        detector_noise_val = np.ones(100)*detector_noise_val
        if len(trace_limits) != 0:
            trace_limits = [int(t/decimation) for t in trace_limits]
        cosy_len = len(self.scan_wavelengths[0:-1]) - len(Cosy[0:-1])

        # save python data
        np_filename = timestr + filename + '.npz'
        np_root = os.path.join(self.transmission_directory_path,np_filename)

        # save PNZ
        np.savez(np_root,
                 spectrum=spectrum_data[0:-1:decimation], wavelengths=self.scan_wavelengths[0:-1:decimation],
                 cosy_spectrum=Cosy[0:-1:decimation], cosy_wavelengths=self.scan_wavelengths[0:-1:decimation],
                 output_wavelength=self.output_wavelength[0:-1:decimation],
                 detector_noise=detector_noise_val, trace_limits=trace_limits)

        return np_root

    # code tests
    def x_axis_resolution_test(self, Waveguide, decimation):
        '''
        this function plot the scan with the limits between scope traces.
        this function isn't necessary in the scan process, use only for checking that the scan precise in X axis
        For changing the resolution in axis X-
        a. change preTriggerSamples & postTriggerSamples in PicoControl_4channels
        b. change the Frequency of SigGen() here in line 48

        Before using this function - make sure SigGen_spectrum collect data from Channel A(input signal to laser)
        '''
        plt.figure()
        plt.title('Transmission Spectrum X axis check')
        plt.xlabel('Wavelength[nm]')
        plt.ylabel('Voltage[mV]')
        plt.grid(True)
        plt.plot(self.scan_wavelengths[0:-1:decimation], self.SigGen_spectrum[0:-1:decimation] / 1000, 'g')
        # plt.plot(self.scan_wavelengths[0:-1:decimation], self.SigGen_spectrum[0:-1:decimation], 'g')
        plt.plot(self.scan_wavelengths[0:-1:decimation], Waveguide[0:-1:decimation], 'orange')

        self.trace_limits = [int(i) for i in self.trace_limits]
        for i in self.trace_limits:
            if i > 50 and i < (len(Waveguide)-50):
                print(i)
                plt.plot(self.scan_wavelengths[i-20: i+20], Waveguide[i-20: i+20], 'y')

        plt.show()
        time.sleep(WAIT_TIME)

    def calibrate_range_test(self):
        '''
        plot all the versions of every trace, for checking the range calibration
        :return:
        '''
        offset_trace = self.Scope.get_trace()[CH_B]
        # add the offset after correction
        fixed_trace = [k + self.Scope.analog_offset for k in offset_trace]

        plt.figure()
        plt.plot(self.Scope.calibrate_trace)
        plt.plot(offset_trace)
        plt.plot(fixed_trace)
        plt.title('Range'+str(list(self.Scope.pico.dictionary_voltage_range.values())[self.Scope.channel_range])+', Peak 2 peak = '+str(self.Scope.trace_range))
        plt.legend(['reference_trace', 'offset_trace', 'fixed_trace'])
        plt.show()


if __name__ == "__main__":
    try:
        Python_Control = False
        o = TransmissionSpectrum(init_wavelength=772, final_wavelength=781, Python_Control=True)
        # get spectrum
        o.get_wide_spectrum_DC_motor_only(parmeters_by_console=True)
        # # this save is for DC_motor_only
        decimation = 1
        o.scan_wavelengths = o.scan_wavelengths[0:o.final_wl_index]
        if len(o.total_Cosy_spectrum) < o.final_wl_index:
            o.save_figure_and_data(r'C:\Users\Lab2\qs-labs\R&D - Lab\Chip Tester\Spectrum_transmission\unnamed scans',o.total_spectrum[0:o.final_wl_index],
                               o.total_Cosy_spectrum[0:o.final_wl_index], decimation, 'Test', detector_noise_val=o.detector_noise, trace_limits=o.trace_limits)
        else:
            o.save_figure_and_data(r'C:\Users\Lab2\qs-labs\R&D - Lab\Chip Tester\Spectrum_transmission\unnamed scans',o.total_spectrum[0:o.final_wl_index],
                               o.total_Cosy_spectrum[0:o.final_wl_index], decimation, 'Test', detector_noise_val=o.detector_noise, trace_limits=o.trace_limits)
        #o.get_wide_spectrum(parmeters_by_console=True)

        # X-axis resolution check: (before use, remove '#' from CH_A lines)
        #o.x_axis_resolution_test(o.total_spectrum, decimation=decimation)

        # o.plot_transmission_spectrum(o.total_spectrum, o.total_Cosy_spectrum, decimation=decimation)
        # o.save_figure_and_data(r'C:\Users\Lab2\qs-labs\R&D - Lab\Chip Tester\Spectrum_transmission\unnamed scans',o.total_spectrum,
        #                        o.total_Cosy_spectrum, decimation, 'Test', detector_noise_val = o.detector_noise, trace_limits = o.trace_limits)
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