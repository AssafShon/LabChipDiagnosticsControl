'''
created at 14/08/22

@author: Assaf S.


'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy import signal
import os
from Utility_functions import bcolors


# PARAMETERS
WAIT_TIME = 1
CH_A = 0
CH_B = 1
CH_C = 2
CH_D = 3


class TransmissionSpectrum:
    def __init__(self,init_wavelength=772, final_wavelength=772.3, Python_Control=True):
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
            # old scan trigger
            self.SigGen = SigGen(pico=self.Pico, pk_to_pk_voltage=0.106, offset_voltage=0, frequency=10, wave_type='PS3000A_TRIANGLE')# frequency = 10
            # new scan trigger ----
            # self.SigGen = SigGen(pico=self.Pico, pk_to_pk_voltage=0.8, offset_voltage=0, frequency=0.033,
            #                    wave_type='PS3000A_SQUARE')  # frequency = 10
            self.Scope = Scope(pico=self.Pico)
            self.Laser = Laser()

            # define variables
            self.final_wavelength = final_wavelength
            self.init_wavelength = init_wavelength
            self.Laser.tlb_set_wavelength(self.init_wavelength)

            # set Laser scan velocity
            self.scan_velocity = 0.5  # nm/sec
            self.Laser.tlb_query('SOURce:WAVE:SLEW:FORWard {}'.format(self.scan_velocity))
            self.Laser.tlb_query('SOURce:WAVE:SLEW:RETurn {}'.format(self.scan_velocity))

            # check if laser TRACK mode is off
            self.track_mode_test()
            # for delete the detector's noise
            self.detector_noise = 0
            try:
                self.detector_noise = 0
                # self.get_detector_noise()
            except Exception:
                raise

    def get_wide_spectrum_DC_motor_only(self,parmeters_by_console):
        '''
        scan wavelength range with laser built-in scan function.
        this way using only DC-monitor without changing the piezo
        :param parmeters_by_console:
        :return:
        '''

        self.total_spectrum = []  # channel B
        self.total_Cosy_spectrum = []   # channel C
        self.SigGen_spectrum = []   # channel A
        self.trace_limits = []
        original_wl = self.init_wavelength

        if parmeters_by_console:

            print("Enter initial wavelength for scan in [nm]:")
            self.init_wavelength = float(input())

            print("Enter final wavelength for scan in [nm]:")
            self.final_wavelength = float(input())

        self.Laser.tlb_query('SOURce:WAVE:START {}'.format(self.init_wavelength-0.5))
        if self.final_wavelength > 781:
            self.final_wavelength = 781
        self.Laser.tlb_query('SOURce:WAVE:STOP {}'.format(self.final_wavelength))
        self.Laser.tlb_set_wavelength(self.init_wavelength-0.5)

        # protect the laser from start scan before set_wavelength(init_wavelength) is done
        if abs(original_wl-(self.init_wavelength-0.5)) > 3:
            time.sleep(abs(original_wl-(self.init_wavelength-0.5))*2*WAIT_TIME+0.5)

        # laser scan
        print('scanning...')
        laser_feedback = self.Laser.tlb_query('OUTPut:SCAN:START')
        if laser_feedback != 'OK':
            raise Exception('Laser scan failed')

        # pico scan
        [self.SigGen_spectrum, self.total_spectrum, self.output_wavelength,
         self.total_Cosy_spectrum] = self.Scope.get_trace(self.init_wavelength, self.final_wavelength)
        print('scan ended. laser still run')

        # remove cosy offset to be around spectrum mean
        self.cosy_offset()
        # get_scan_wavelengths
        self.get_scan_wavelengths()

    def get_detector_noise(self):
        '''
        turn off the laser, find the constant noise of the detector and turn on the laser
        detector_noise = mean value of pico trace with the laser off
        :return:
        '''
        # precaution - run the function only if the laser turned on by the user
        if self.Laser.tlb_query('OUTPut:STATe?') != '1':
            print('laser error - laser is off:)')
            raise Exception('Error. Turn on the laser and try again')

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
                print('laser error - didn\'t turned on well')
                raise Exception('Error in turning on the laser')
        else:
            print('laser return output state \' \', waiting more 5 sec and checks again')
            time.sleep(WAIT_TIME * 5)
            if self.Laser.tlb_query('OUTPut:STATe?') != '0':
                # understanding the laser error
                print('laser error - after turning off the laser \'output state\' != 0')
                laser_reaction = self.Laser.tlb_query('OUTPut:STATe?')
                print('after turning off, laser said = '+str(laser_reaction))
                raise Exception('laser error - after turning off the laser \'output state\' != 0')

        time.sleep(WAIT_TIME * 5)
        laser_reaction = self.Laser.tlb_query('OUTPut:STATe?')
        if laser_reaction == '1':
            print('The laser turned on, the noise is '+str(np.round(self.detector_noise, 3)))
        else:
            print('after turning on, laser said = ' + str(laser_reaction))
            raise Exception('laser error - after turning on the laser \'output state\' != 1')

    def cosy_offset(self):
        '''
        remove CoSy offset to be around the spectrum mean value
        :return:
        '''
        # offset to be around zero
        self.total_Cosy_spectrum = self.total_Cosy_spectrum - np.mean(self.total_Cosy_spectrum)
        # offset to be around total_spectrum mean value
        self.total_Cosy_spectrum = self.total_Cosy_spectrum + np.mean(self.total_spectrum)
        # flip CoSy in y-axis
        self.total_Cosy_spectrum = -self.total_Cosy_spectrum

    def get_scan_wavelengths(self):
        '''
        find the final wavelength by output wavelength data.
        final_wl_index = index of self.final_wavelength
        scan_wavelengths = vector of spectrum wavelengths
        :return:
        '''
        first_final_index = self.output_wavelength.index(np.max(self.output_wavelength))  # first max value
        reversed_output_wl = self.output_wavelength[first_final_index:first_final_index+499]
        reversed_output_wl.reverse()
        temp_index = reversed_output_wl.index(np.max(reversed_output_wl))
        last_final_index = first_final_index + len(reversed_output_wl) - temp_index - 1  # last max value
        self.final_wl_index = int((last_final_index + first_final_index)/2)

        sample_in_nm = (self.final_wavelength - self.init_wavelength) / self.final_wl_index
        self.scan_wavelengths = (sample_in_nm * np.arange(0, len(self.total_spectrum)))+self.init_wavelength

    def plot_transmission_spectrum(self, decimation=1):
        # make output_wavelength smaller
        self.output_wavelength = [j/100 for j in self.output_wavelength]

        # plot
        plt.figure()
        plt.title('Transmission Spectrum')
        plt.xlabel('Wavelength[nm]')
        plt.ylabel('Voltage[mV]')
        plt.legend(['full scan range', 'output_wavelength', 'SigGen_spectrum', 'cosy'])
        # all data
        plt.plot(self.scan_wavelengths[0:-1:decimation], self.total_spectrum[0:-1:decimation], 'r')
        plt.plot(self.scan_wavelengths[0:-1:decimation], self.output_wavelength[0:-1:decimation], 'lightblue')
        # plt.plot(self.scan_wavelengths[0:-1:decimation], self.SigGen_spectrum[0:-1:decimation], 'b')
        plt.plot(self.scan_wavelengths[0:-1:decimation], self.total_Cosy_spectrum[0:-1:decimation], 'g')
        # only [init_wavelength:final_wavelength]
        plt.plot(self.scan_wavelengths[0:self.final_wl_index:decimation], self.total_spectrum[0:self.final_wl_index:decimation], 'y')
        plt.show()

    def track_mode_test(self):
        # check that Track mode of the laser is off
        # time.sleep(WAIT_TIME*(self.final_wavelength - self.init_wavelength - 0.5)*2 + 0.5)
        if self.Laser.tlb_query('OUTPut:TRACK?') == '0':
            print('Track mode is off')
        else:
            print(bcolors.WARNING + 'Track mode is on' + bcolors.ENDC)

    def save_figure_and_data(self, dist_root, spectrum_data, Cosy, decimation, filename='Transmission_spectrum',
                             mkdir=True, detector_noise_val=12, trace_limits=[]):
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
                 SigGen_spectrum=self.SigGen_spectrum[0:-1:decimation],
                 output_wavelength=self.output_wavelength[0:-1:decimation],
                 detector_noise=detector_noise_val, trace_limits=trace_limits)

        return np_root

if __name__ == "__main__":
    try:
        Python_Control = False
        o = TransmissionSpectrum(init_wavelength=769.5, final_wavelength=781, Python_Control=True)
        # get spectrum
        o.get_wide_spectrum_DC_motor_only(parmeters_by_console=True)
        decimation = 1
        o.plot_transmission_spectrum(decimation)
        o.scan_wavelengths = o.scan_wavelengths[0:o.final_wl_index]
        # if len(o.total_Cosy_spectrum) < o.final_wl_index:
        o.save_figure_and_data(r'C:\Users\Lab2\qs-labs\R&D - Lab\Chip Tester\Spectrum_transmission\unnamed scans',o.total_spectrum[0:o.final_wl_index],
                        o.total_Cosy_spectrum[0:o.final_wl_index], decimation, 'Test', detector_noise_val=o.detector_noise, trace_limits=o.trace_limits)
        # else:
        #     o.save_figure_and_data(r'C:\Users\Lab2\qs-labs\R&D - Lab\Chip Tester\Spectrum_transmission\unnamed scans',o.total_spectrum[0:o.final_wl_index],
        #                        o.total_Cosy_spectrum[0:o.final_wl_index], decimation, 'Test', detector_noise_val=o.detector_noise, trace_limits=o.trace_limits)
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