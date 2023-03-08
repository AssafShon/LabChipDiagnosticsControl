
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
        self.saved_file_root = r'C:\Users\Lab2\qs-labs\R&D - Lab\Chip Tester\Spectrum_transmission\Statstic'
        # load_filename = r'20230110-102329Test.npz'


        chip_types_names = os.listdir(self.saved_file_root)

        self.chips_dictionary = {}
        self.excel_contant = ""

        #for elem in chip_types_names:
        for elem in ["01A3"]:    # fast version for debug
            single_chip_type_name = os.path.join(self.saved_file_root, elem)
            chip_number_names = os.listdir(single_chip_type_name)

            for chip in chip_number_names:
            #for chip in ["chip2"]:     # fast version for debug
                waveguides_dictionary_data = self.all_wg_of_singel_chip(dist_root_=os.path.join(single_chip_type_name, chip))
                self.chips_dictionary[chip, elem] = waveguides_dictionary_data
                self.excel_contant = self.excel_contant+elem+"-"+chip+","

        self.save_statistics()

    def all_wg_of_singel_chip(self, dist_root_):
        self.waveguides_dictionary = {}

        self.wg_names = []
        for i in range(11,13):
            for j in range(1, 2):
                self.wg_names.append(r'W'+str(j)+'-'+str(i).zfill(2))
        for name in self.wg_names:
            analysis_path = dist_root_ + '\\' + name
            print(analysis_path[-16:]+":")
            re_analyze = 0
            skip = 0
            while re_analyze == 0:
                if not os.path.isdir(analysis_path):
                    print('This WG scan don\'t exist')
                    skip = 1
                    break
                super().__init__(run_experiment="false", saved_file_root=analysis_path)
                # print("Is the result okay? [0-No, return  1-Yes, move on]")
                # re_analyze = int(input())
                re_analyze = 1
            plt.close('all')
            if skip == 0:
                self.waveguides_dictionary[name] = self.analysis_spectrum_parameters




        return self.waveguides_dictionary

    def save_statistics(self):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        # create excel file
        analysis_path_with_time = os.path.join(r'C:\Users\Lab2\qs-labs\R&D - Lab\Chip Tester\Spectrum_transmission\Statistic_scans_results',self.excel_contant+timestr)
        prameters_csv = os.path.join(analysis_path_with_time+'.csv')
        with open(prameters_csv, 'w') as f:
            f.write("File Contant:,")
            f.write("%s\n\n" % self.excel_contant)

            f.write("%s,%s,%s,%s,%s,%s,%s,%s, %s \n" % ("mode[THz]","peak_freq[GHz]","kappa_ex[GHz]","kappa_i[GHz]","h","FWHM[GHz]","wave_guide","chip number", "wayfer"))
            for chip_name in self.chips_dictionary.keys():
                for wg_name in self.chips_dictionary[chip_name]:
                    for i in range(len(self.chips_dictionary[chip_name][wg_name]["mode[THz]"])):
                        for wg_character in self.chips_dictionary[chip_name][wg_name]:
                            f.write("%s," % self.chips_dictionary[chip_name][wg_name][wg_character][i])
                        f.write("%s, %s, _%s\n" % (wg_name, chip_name[0], chip_name[1]))


if __name__ == "__main__":
    o = StatisticAnalyze(decimation=1)



