from TransmissionSpectrum import TransmissionSpectrum
import numpy as np
import time
from BasicInstrumentsControl.KeithleyPwrSupplyControl.KeithleyPwrSupplyControl import KeithleyPwrSupplyControl as PowerSupply
from AnalyzeSpectrum import AnalyzeSpectrum
from numpy import matlib
from matplotlib import pyplot as plt
from Utility_functions import plot_figure
WAIT_TIME = 5


class HeaterScan(TransmissionSpectrum):
    def __init__(self, max_current_scan=10e-3, num_of_points_in_scan=5, typ_noise_in_freq = 20e-3,
                 decimation=1000,division_width_between_modes = 8.0e-3 , saved_file_root =  r'C:\Users\Lab2\qs-labs\R&D - Lab\Chip Tester\HeaterScan'):
        # running transmission spectrum init (connect pico and laser)
        super().__init__()
        # connect pwr supply
        self.PowerSupply = PowerSupply()

        # define properties
        self.max_current_scan = max_current_scan # [A]
        self.num_of_points_in_scan = num_of_points_in_scan
        self.typ_noise_in_freq = typ_noise_in_freq # [THz]
        self.decimation = decimation
        self.division_width_between_modes = division_width_between_modes
        self.saved_file_root = saved_file_root

        self.PowerSupply.SetCurrent(0)
        self.PowerSupply.SetVoltage(20) # so voltage won't restrict the current
        self.PowerSupply.OutputState(1)

        self.scan_current_to_heater()

        self.PowerSupply.OutputState(0)
        self.PowerSupply.Disconnect()

    def scan_current_to_heater(self):
        '''
        scan the current from the current supply, takes trace with pico, detects the resonance peak and plots
         the peak wavelength as function of current. Must restrict the current to 10mA.
        :return:
        '''
        # define variables
        self.all_peaks = []
        self.spectrum_per_current = []

        self.currents_in_scan = np.arange(0, self.max_current_scan, self.max_current_scan/self.num_of_points_in_scan)
        for idx, current in enumerate(self.currents_in_scan):
            # set current to power supply
            self.PowerSupply.SetCurrent(current)   # set current by voltage

            # get trace from scope and detect resonance center
            print("begin "+str(idx)+" scan")
            if idx==0:
                mkdir=True
                self.get_wide_spectrum(parmeters_by_console=True)
                fig_peaks_colored, ax_peaks_colored = plt.subplots()

            else:
                mkdir=False
                time.sleep(WAIT_TIME)
                self.get_wide_spectrum(parmeters_by_console=False)

            self.spectrum_per_current.append(self.total_spectrum)
            # self.save_figure_and_data(self.saved_file_root,
            #                        self.total_spectrum, 1000, 'Test',mkdir)
            peaks = self.analyze_spectrum(self.total_spectrum,idx,current,ax=ax_peaks_colored)
            self.all_peaks.append(self.scan_freqs[peaks])
        #
        # self.heated_peak = self.find_heated_peak()
        # # pin heated peaks on spectrum
        # for i in range(len(self.spectrum_per_current)):
        #     AnalyzeSpectrum.plot_peaks(scan_freqs=self.scan_freqs, interpolated_spectrum=self.spectrum_per_current[i][:-1],
        #                            peaks_per_mode=sum(self.heated_peaks_ind,[]))
        # # plot heated peak freq as function of current
        # plt.figure()
        # for i in range(len(self.heated_peak[0])):
        #     plt.plot(self.heated_peak,self.currents_in_scan,label = 'peak '+str(i))
        #     plt.legend()

    def analyze_spectrum(self,spectrum,i,current,ax):
        '''

        :param spectrum:
        :param i:
        :return: only the fundamental mode peaks.
        '''
        # convert from nm to THz
        self.scan_freqs = AnalyzeSpectrum.get_scan_freqs(scan_wavelengths=self.scan_wavelengths)

        # smooth spectrum
        interpolated_spectrum = AnalyzeSpectrum.smooth_and_normalize_spectrum(decimation=self.decimation, spectrum=spectrum,
                                                                              wavelengths=self.scan_wavelengths)
        peaks_width, peaks, peaks_properties = AnalyzeSpectrum.find_peaks_in_spectrum( prominence=0.2, height=None, distance=None, rel_height=0.5,spectrum=interpolated_spectrum[0])
        peaks_width_in_Thz = [self.scan_freqs[( peaks[i] + int(peaks_width[0][i] / 2))] -
                                   self.scan_freqs[(peaks[i] - int(peaks_width[0][i] / 2))] for i in
                                   range(len(peaks))]
        # plot peaks colored by widths
        AnalyzeSpectrum.plot_peaks_colored_by_width(ax=ax,peaks_freqs=self.scan_freqs[peaks],Y=[current]*len(peaks),colors=peaks_width[0])

        # divide different modes
        fundamental_mode,high_mode = AnalyzeSpectrum.divide_to_different_modes(peaks=peaks,
            division_width_between_modes=self.division_width_between_modes,
                                       modes_width=peaks_width_in_Thz)
        [peaks_fundamental_mode, peaks_high_mode] = [fundamental_mode[1], high_mode[1]]
        self.peaks_per_mode = [peaks_fundamental_mode, peaks_high_mode]
        peaks_fig=AnalyzeSpectrum.plot_peaks(scan_freqs=self.scan_freqs,interpolated_spectrum=interpolated_spectrum[0],
                                   peaks_per_mode=self.peaks_per_mode)


        AnalyzeSpectrum.save_analyzed_data(dist_root=self.saved_file_root,
                                           filename=r'Heater_Scan'+str(i)
                                , analysis_spectrum_parameters=None,
                                spectrum_data=[interpolated_spectrum, self.scan_freqs],figure=peaks_fig, width_fig = fig_peaks_colored)
        return peaks_fundamental_mode




    def find_heated_peak(self):
        '''
        Returns heated peak for different currents. This is done by binning the peaks location (to take care of noise in
         peak location) filtering the constant bins and extracting out the heated peak location.
        :return:
        '''
        freqs_range = (self.scan_freqs[-1],self.scan_freqs[0])
        freqs_range_len = self.scan_freqs[0] - self.scan_freqs[-1]
        num_of_bins = int(freqs_range_len/self.typ_noise_in_freq) # the bin width will be approx. typ_noise_in_freq
        binned_peaks = np.zeros((len(self.all_peaks),num_of_bins))
        for i in range(len(self.all_peaks)):
            binned_peaks[i] = np.histogram(a=self.all_peaks[i], bins=num_of_bins,range=freqs_range)[0] # histogram bin - init<=data<final
        binned_peaks=(binned_peaks>0)*1
        # Decreasing the constant bins by decreasing the product of the lines (only all 1's will ber decreased)
        binned_peaks_filtered = binned_peaks - matlib.repmat(np.prod(binned_peaks, axis=0),len(binned_peaks),1)
        heated_peak = self.extract_heated_peak_locations(binned_peaks_filtered)
        return heated_peak

    def extract_heated_peak_locations(self,binned_peaks_filtered):
        '''
        ind_binned_heated_peak  - the index in bining of the heated peak.
        :param binned_peaks_filtered:
        :return:
        '''
        # nonzero() returns tuple of nonzeros in rows ([0])  and columns ([1])
        non_zero_binned_peaks_filtered = np.nonzero(binned_peaks_filtered)
        heated_peak = [[] for _ in self.all_peaks]
        self.heated_peaks_ind = [[] for _ in self.all_peaks]

        # run over all-peaks and find for each row the element that is related to the non-zero binning indices
        for i in range(len(self.all_peaks)):
            bin_number_heated_peak = non_zero_binned_peaks_filtered[1][np.where(non_zero_binned_peaks_filtered[0] == 0)]
            for j in range(len(self.all_peaks[i])):
                if int((self.all_peaks[i][j]-self.scan_freqs[-1])//self.typ_noise_in_freq) in bin_number_heated_peak:
                    heated_peak[i].append(self.all_peaks[i][j])
                    self.heated_peaks_ind[i].append(np.where(self.scan_freqs == self.all_peaks[i][j]))
        return heated_peak

if __name__ == "__main__":
    o = HeaterScan()
    # plt.plot(np.arange(0, o.max_current_scan, o.num_of_points_in_scan), o.resonance_center)
