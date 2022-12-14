from TransmissionSpectrum import TransmissionSpectrum
import numpy as np
import time
from BasicInstrumentsControl.KeithleyPwrSupplyControl.KeithleyPwrSupplyControl import KeithleyPwrSupplyControl as PowerSupply
from AnalyzeSpectrum import AnalyzeSpectrum

WAIT_TIME = 0.1


class HeaterScan(TransmissionSpectrum):
    def __init__(self, max_current_scan=10e-3, num_of_points_in_scan=5, typ_noise_in_freq = 3e-3,
                 decimation=1000,division_width_between_modes = 5.0e-3):
        super().__init__()
        self.PowerSupply = PowerSupply()

        self.max_current_scan = max_current_scan # [A]
        self.num_of_points_in_scan = num_of_points_in_scan
        self.typ_noise_in_freq = typ_noise_in_freq
        self.decimation = decimation
        self.division_width_between_modes = division_width_between_modes

        self.PowerSupply.SetVoltage(30) # so voltage won't restrict the current
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

        for idx, current in enumerate(np.arange(0, self.max_current_scan, self.num_of_points_in_scan)):
            # set current to power supply
            self.PowerSupply.SetCurrent(current)   # set current by voltage
            time.sleep(WAIT_TIME)

            # get trace from scope and detect resonance center
            if idx==0:
                self.get_wide_spectrum(parmeters_by_console=True)
                # self.get_wide_spectrum(parmeters_by_console=True)
            else:
                self.get_wide_spectrum(parmeters_by_console=False)
            peaks = self.analyze_spectrum(self.total_spectrum,idx)
            self.all_peaks.append(self.total_spectrum[peaks])

        self.heated_peak = self.find_heated_peak()

    def analyze_spectrum(self,spectrum,i):
        # convert from nm to THz
        self.scan_freqs = AnalyzeSpectrum.get_scan_freqs(scan_wavelengths=self.scan_wavelengths)

        # smooth spectrum
        interpolated_spectrum = AnalyzeSpectrum.smooth_spectrum(decimation=self.decimation, spectrum=spectrum,
                                                                wavelengths=self.scan_wavelengths)
        peaks_width, peaks, peaks_properties = AnalyzeSpectrum.find_peaks_in_spectrum( prominence=15, height=None, distance=None, rel_height=0.5,spectrum=interpolated_spectrum)
        peaks_width_in_Thz = [self.scan_freqs[( peaks[i] - int(peaks_width[0][i] / 2))] -
                                   self.scan_freqs[(peaks[i] + int(peaks_width[0][i] / 2))] for i in
                                   range(len(peaks))]

        # divide different modes
        fundamental_mode,high_mode = AnalyzeSpectrum.divide_to_different_modes(peaks=peaks,
            division_width_between_modes=self.division_width_between_modes,
                                       modes_width=peaks_width_in_Thz)
        [peaks_fundamental_mode, peaks_high_mode] = [fundamental_mode[1], high_mode[1]]
        self.peaks_per_mode = [peaks_fundamental_mode, peaks_high_mode]
        AnalyzeSpectrum.plot_peaks(scan_freqs=self.scan_freqs,interpolated_spectrum=interpolated_spectrum,
                                   peaks_per_mode=self.peaks_per_mode)
        AnalyzeSpectrum.save_analyzed_data(dist_root=r'C:\Users\Lab2\qs-labs\R&D - Lab\Chip Tester\HeaterScan',
                                           filename=r'Heater_Scan'+str(i)
                                , analysis_spectrum_parameters=None,
                                spectrum_data=[interpolated_spectrum, self.scan_freqs])
        return peaks_fundamental_mode




    def find_heated_peak(self):
        '''
        Returns heated peak for different currents. This is done by binning the peaks location (to take care of noise in
         peak location) filtering the constant bins and extracting out the heated peak location.
        :return:
        '''
        freqs_range = (self.scan_freqs[0],self.scan_freqs[-1])
        freqs_range_len = self.scan_freqs[-1] - self.scan_freqs[0]
        num_of_bins = int(freqs_range_len/self.typ_noise_in_freq) # the bin width will be approx. typ_noise_in_freq
        binned_peaks = np.zeros((len(self.all_peaks),num_of_bins))
        for i in range(len(self.all_peaks)):
            binned_peaks[i] = np.histogram(a=self.all_peaks[i], bins=num_of_bins,range=freqs_range)[0] # histogram bin - init<=data<final

        # Decreasing the constant bins by decreasing the product of the lines (only all 1's will ber decreased)
        binned_peaks_filtered = binned_peaks - np.prod(binned_peaks, axis=0)
        heated_peak = self.extract_heated_peak_locations(binned_peaks_filtered)
        return heated_peak

    def extract_heated_peak_locations(self,binned_peaks_filtered):
        '''
        ind_binned_heated_peak  - the index in bining of the heated peak.
        :param binned_peaks_filtered:
        :return:
        '''
        ind_binned_heated_peak = np.nonzero(binned_peaks_filtered)[1] # nonzero()[1] is the array of nonzeros in each row
        heated_peak = np.zeros(len(self.all_peaks))

        # run over all-peaks and fid for each row the element that is related to the non-zero binning indices
        for i in range(len(self.all_peaks)):
            for j in range(len(self.all_peaks[i])):
                if (self.all_peaks[i][j]-self.scan_freqs[0])//self.typ_noise_in_freq == ind_binned_heated_peak[i]:
                    heated_peak[i]=self.all_peaks[i][j]
        return heated_peak

if __name__ == "__main__":
    o = HeaterScan()
    # plt.plot(np.arange(0, o.max_current_scan, o.num_of_points_in_scan), o.resonance_center)
