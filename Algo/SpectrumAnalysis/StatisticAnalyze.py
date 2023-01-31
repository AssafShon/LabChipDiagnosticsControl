
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

        # print("what is the full path of the chip's data?")
        # saved_file_root = input()
        self.saved_file_root = r'C:\Users\asafs\qs-labs\R&D - Lab\Chip Tester\Spectrum_transmission\Statstic'
        # load_filename = r'20230110-102329Test.npz'

        chip_types_names = os.listdir(self.saved_file_root)

        self.chips_dictionary = {}

        for elem in chip_types_names:
            single_chip_type_name = os.path.join(self.saved_file_root, elem)
            chip_number_names = os.listdir(single_chip_type_name)

            for chip in chip_number_names:
                waveguides_dictionary_data = self.all_wg_of_singel_chip(dist_root_=os.path.join(single_chip_type_name, chip))
                self.chips_dictionary[chip] = waveguides_dictionary_data

        self.save_statistics()

    def all_wg_of_singel_chip(self, dist_root_):
        waveguides_dictionary = {}

        self.wg_names = []
        for i in range(9, 11):
            for j in range(1,2):
                self.wg_names.append(r'\W'+str(j)+'-'+str(i).zfill(2))
        for name in self.wg_names:
            analysis_path = dist_root_ + name
            super().__init__(run_experiment="false", saved_file_root = analysis_path)
            plt.close('all')
            waveguides_dictionary[name] = self.analysis_spectrum_parameters

        return waveguides_dictionary

    def save_statistics(self):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        # create directory
        analysis_path_with_time = os.path.join(r'C:\Users\asafs\qs-labs\R&D - Lab\Chip Tester\Spectrum_transmission\statistic_scans_results',timestr)
        prameters_csv = os.path.join(analysis_path_with_time+'.csv')
        with open(prameters_csv, 'w') as f:
            for key in self.chips_dictionary.keys():
                f.write("%s,%s\n" % (key, self.chips_dictionary[key]))
                for in_key in self.chips_dictionary[key]:
                    f.write("%s,%s\n, %s\n" % (key, in_key, self.chips_dictionary[key][in_key]))

if __name__ == "__main__":
    o = StatisticAnalyze(decimation=1)



