from TransmissionSpectrum import TransmissionSpectrum
import numpy as np
import time
from BasicInstrumentsControl.KeithleyPwrSupplyControl import KeithleyPwrSupplyControl as PowerSupply
# from AnalyzeSpectrum import find_peaks_in_spectrum, smooth_spectrum
from AnalyzeSpectrum import AnalyzeSpectrum

WAIT_TIME = 0.1


class HeaterScan(TransmissionSpectrum):
    def __init__(self, resonance_wavelength=776, max_current_scan=5, num_of_points_in_scan=5, init_scan_freq = 774,final_scan_freq = 774, typ_noise_in_freq = 3e-3):
        super().__init__()
        self.PowerSupply = PowerSupply()

        self.resonance_wavelength = resonance_wavelength
        self.max_current_scan = max_current_scan
        self.num_of_points_in_scan = num_of_points_in_scan
        self.typ_noise_in_freq = typ_noise_in_freq

    def scan_current_to_heater(self):
        '''
        scan the current from the current supply, takes trace with pico, detects the resonance peak and plots
         the peak wavelength as function of current. Must restrict the current to 10mA.
        :return:
        '''
        # define variables
        self.resonance_center = np.zeros(1, self.num_of_points_in_scan)
        self.all_peaks = []

        self.Laser.tlb_set_wavelength(self.resonance_wavelength)

        for idx, current in enumerate(np.arange(0, self.max_current_scan, self.num_of_points_in_scan)):
            # set current to power supply
            self.PowerSupply.set_current(current)   # set current by voltage
            time.sleep(WAIT_TIME)

            # get trace from scope and detect resonance center
            spectrum = self.Scope.get_trace()
            interpolated_spectrum = AnalyzeSpectrum.smooth_spectrum(decimation=1000, spectrum=spectrum,
                                                    wavelengths=self.scan_wavelengths)
            _,self.peaks = AnalyzeSpectrum.find_peaks_in_spectrum(interpolated_spectrum)
            self.all_peaks.append(self.peaks)

            self.heated_peak = self.find_heated_peak()


            self.resonance_center[idx]
    def find_heated_peak(self):
        binned_peaks = np.histogram(a=self.all_peaks,bins=self.typ_noise_in_freq)
        binned_peaks_filtered = binned_peaks - np.prod(binned_peaks, axis=1)
        heated_peak = self.extract_heated_peak_locations(binned_peaks_filtered)
        return heated_peak

    def set_binning(self,data, bin_size):
        for i in range()

if __name__ == "__main__":
    o = HeaterScan()
    o.scan_current_to_heater()
    # plt.plot(np.arange(0, o.max_current_scan, o.num_of_points_in_scan), o.resonance_center)
