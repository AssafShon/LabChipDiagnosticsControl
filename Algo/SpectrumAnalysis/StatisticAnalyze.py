
from scipy.interpolate import CubicSpline
from scipy.signal import peak_widths, find_peaks
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from TransmissionSpectrum import TransmissionSpectrum
import os
import time
from AnalyzeSpectrum import AnalyzeSpectrum
import csv
from Utility_functions import bcolors


class StatisticAnalyze(AnalyzeSpectrum):
    def __init__(self, decimation=1):
        '''
        for every waveguide in a chip folder - take the excel results for every waveguide and create one excel file for the chip.
        use AnalyzeSpectrum class for analyzing.

        :param decimation: defines the decimation on the signal at the function smooth spectrum (takes each "decimation" index from the spectrum and then interpulates between the points)
        :param prominence: The prominence of a peak measures how much a peak stands out from the surrounding
                           baseline of the signal and is defined as the vertical distance between the peak and its lowest contour line.:
        :param height: Required height of peaks. Either a number, None, an array matching x or a 2-element squence of thee former.
                       The first element is always interpreted as the minimal and the second, if supplied, as the maximal required height.
        :param distance: Required minimal horizontal distance (>= 1) in samples between neighbouring peaks. Smaller peaks are removed first until the condition is fulfilled for all remaining peaks.
        :param rel_height:Used for calculation of the peaks width, thus it is only used if width is given. See argument rel_height in peak_widths for a full description of its effects.
        :param run_experiment: bool between running an experiment via Analyze spectrum or just analyzing loaded data.
        :param division_width_between_modes:the frequency whoch divides the modes [Thz]
        '''
        # :param max_diff_between_widths_coeff:the maximum difference between widths of modes to be considered as a group of same mode
        print("what is the full path of the chip's data?")
        saved_file_root = input()
        # saved_file_root = r'C:\Users\Lab2\qs-labs\R&D - Lab\Chip Tester\Spectrum_transmission\Statstic\07E0\chip2\W1-09'
        # load_filename = r'20230110-102329Test.npz'


        self.collect_all_waveguides_data(dist_root_=saved_file_root)
        self.chip_analysis_spectrum_parameters = {}

    @classmethod
    def collect_all_waveguides_data(self, dist_root_):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        total_chip_parameters_csv = os.path.join(dist_root_, 'total_chip_parameters_' + timestr + '.csv')

        for i in range(3, 4):
            analysis_path = dist_root_ + r'\W1-0' + str(i);
            if os.path.isdir(analysis_path):
                pnz_files_in_folder = [file for file in os.listdir(analysis_path) if file.endswith('.npz')]
                np_root = os.path.join(analysis_path, pnz_files_in_folder[0])
                # np_root = os.path.join(self.transmission_directory_path, load_filename)

                data = np.load(np_root)
                self.total_spectrum = data['spectrum']
                self.scan_wavelengths = data['wavelengths']
                self.analysing_using_analyzeSpectrum()
                self.chip_analysis_spectrum_parameters['W1-0'+str(i)] = [self.analysis_spectrum_parameters]

        for i in range(3, 3):
            analysis_path = dist_root_ + r'\W2-0' + str(i);
            if os.path.isdir(analysis_path):
                pnz_files_in_folder = [file for file in os.listdir(analysis_path) if file.endswith('.npz')]
                np_root = os.path.join(analysis_path, pnz_files_in_folder[0])

                data = np.load(np_root)
                self.total_spectrum = data['spectrum']
                self.scan_wavelengths = data['wavelengths']
                self.analysing_using_analyzeSpectrum()
                self.chip_analysis_spectrum_parameters['W2-0' + str(i)] = [self.analysis_spectrum_parameters]

        with open(total_chip_parameters_csv, 'w') as f:
            for key in self.chip_analysis_spectrum_parameters.keys():
                f.write("%s,%s\n" % (key, self.chip_analysis_spectrum_parameters[key]))

    @classmethod
    def analysing_using_analyzeSpectrum(self, decimation=1, prominence=0.1, height=None, distance=None, rel_height=0.5, division_width_between_modes=3e-3,
                                        fsr=1.3, num_of_rings=4, init_frequency=384, diff_between_groups=0.03):

        # convert from nm to THz
        self.scan_freqs = self.get_scan_freqs(self.scan_wavelengths)

        # smooth spectrum and normalize it
        self.interpolated_spectrum = self.smooth_and_normalize_spectrum(decimation=1,
                                                                        spectrum=self.total_spectrum,
                                                                        wavelengths=self.scan_wavelengths)

        # find peaks and divide to different modes
        self.peaks_width, self.peaks, self.peaks_properties = self.find_peaks_in_spectrum(prominence, height,
                                                                                          distance, rel_height,
                                                                                          spectrum=self.interpolated_spectrum)
        self.peaks_width_in_Thz = [self.scan_freqs[(self.peaks[i] - int(self.peaks_width[0][i] / 2))] -
                                   self.scan_freqs[(self.peaks[i] + int(self.peaks_width[0][i] / 2))] for i in
                                   range(len(self.peaks))]

        # divide different modes
        self.fundamental_mode, self.high_mode = self.divide_to_different_modes(peaks=self.peaks,
                                                                               division_width_between_modes=division_width_between_modes,
                                                                               modes_width=self.peaks_width_in_Thz)
        [self.peaks_fundamental_mode, self.peaks_high_mode] = [self.fundamental_mode[1], self.high_mode[1]]
        self.peaks_fund_mode_ind = self.fundamental_mode[0]
        self.peaks_per_mode = [self.peaks_fundamental_mode, self.peaks_high_mode]

        # plot figure with peaks
        self.plot_peaks(scan_freqs=self.scan_freqs, interpolated_spectrum=self.interpolated_spectrum,
                        peaks_per_mode=self.peaks_per_mode)

        # fit lorenzians
        [self.fit_res, self.fit_cov_params] = self.fit_lorenzians(self)

        #
        self.effective_kappa_all_resonances = self.calc_effective_kappa_and_h()

        # plot lorenzians
        self.plot_lorenzians()
        plt.show()

        # classify peaks to different rings

        self.classify_peaks(fsr, num_of_rings, init_frequency, diff_between_groups)

        # get parameters and save them
        self.get_analysis_spectrum_parameters()


if __name__ == "__main__":
    o = StatisticAnalyze(decimation=1)




# save figure
        plt.savefig(os.path.join(analysis_path, timestr + filename + '.png'))
        # save data as csv
        if not analysis_spectrum_parameters is None:
            prameters_csv = os.path.join(analysis_path, 'parameters_' + timestr + filename + '.csv')
            with open(prameters_csv, 'w') as f:
                for key in analysis_spectrum_parameters.keys():
                    f.write("%s,%s\n" % (key, analysis_spectrum_parameters[key]))
            # save python data
            np.savez(os.path.join(analysis_path, 'parameters_' + timestr + filename + '.npz'),
                     parameters=analysis_spectrum_parameters)

        # save figure data in python
        np.savez(os.path.join(analysis_path, 'spectrum_data_' + timestr + filename + '.npz'),
                 spectrum=spectrum_data)