
from scipy.interpolate import CubicSpline
from scipy.signal import peak_widths, find_peaks
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from TransmissionSpectrum import TransmissionSpectrum
import math
import cluster

#parameters
C_light= 2.99792458e8

class AnalyzeSpectrum(TransmissionSpectrum):
    def __init__(self, decimation=1, prominence=20, height=None, distance=None, rel_height=0.5,
                 run_experiment=False,division_width_between_modes = 2.5e-3):
        '''

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
        #:param max_diff_between_widths_coeff:the maximum difference between widths of modes to be considered as a group of same mode
        if run_experiment:
            super().__init__()
            pass
        else:
            data = np.load(
                r'C:\Users\asafs\qs-labs\R&D - Lab\Chip Tester\Spectrum_transmission\Statstic\01A3\chip1\W1-09\20221207-104842Test.npz')
            self.total_spectrum = data['spectrum']
            self.scan_wavelengths = data['wavelengths']

        # convert from nm to THz
        self.scan_freqs = self.get_scan_freqs()

        # smooth spectrum
        self.interpolated_spectrum = self.smooth_spectrum(decimation, spectrum =self.total_spectrum, wavelengths=self.scan_wavelengths)

        # find peaks and divide to different modes
        self.peaks_width,self.peaks,self.peaks_properties = self.find_peaks_in_spectrum(prominence,height,distance,rel_height,spectrum=self.interpolated_spectrum)

        #plot figure with peaks
        self.plot_peaks()

        #fit lorenzians
        self.fit_res = self.fit_lorenzians()
        self.effective_kappa_all_resonances = self.calc_effective_kappa()

        #divide different modes
        self.divide_to_different_modes(division_width_between_modes = division_width_between_modes,modes_width = self.effective_kappa_all_resonances)

        #


        #plot lorenzians
        self.plot_lorenzians()
        plt.show()

    def classify_peaks(self,fsr, num_of_rings):
        '''
        classify peaks to their fsr and ring number
        :param fsr - the distance between peaks
        :param num of rings - number of rings
        :return:
        '''


    def divide_to_different_modes(self,modes_width, division_width_between_modes):  # max_diff_between_widths_coeff=0.1):
        '''
        divides peaks into different modes depending on their width
        :param diff_condition_between_modes_width - defines the difference in mode width to be considered as the same mode
        :return:
        '''
        # cl = cluster.HierarchicalClustering(modes_width, lambda x, y: abs(x - y))
        # self.peaks_width_per_mode = cl.getlevel(max_diff_between_widths_coeff*np.mean(modes_width)) fundamental

        self.widths_fundamental_mode = [a for a in modes_width if a<division_width_between_modes]
        self.peaks_fund_mode_ind = [j for j, x in enumerate(modes_width) if x in self.widths_fundamental_mode]
        self.peaks_fundamental_mode = [self.peaks[k] for k in self.peaks_fund_mode_ind]


        self.widths_high_mode = [a for a in modes_width if a>division_width_between_modes]
        self.peaks_high_mode_ind = [j for j, x in enumerate(modes_width) if x in self.widths_high_mode]
        self.peaks_fundamental_mode = [self.peaks[k] for k in self.peaks_high_mode_ind]


    def plot_peaks(self):
        plt.figure()
        plt.plot(self.scan_freqs, self.interpolated_spectrum)
        plt.plot(self.scan_freqs[self.peaks], self.interpolated_spectrum[self.peaks], 'o')
        plt.hlines(self.peaks_width[1], self.scan_freqs[(self.peaks_width[2]).astype(int)],
                   self.scan_freqs[(self.peaks_width[3]).astype(int)])
        # plt.plot(self.scan_freqs[(self.peaks[4] - int(self.peaks_width[0][4]*0.7)):(self.peaks[4] + int(self.peaks_width[0][4]*0.7))],
        #          self.interpolated_spectrum[(self.peaks[4] - int(self.peaks_width[0][4]*0.7)):(self.peaks[4] + int(self.peaks_width[0][4]*0.7))])

    def plot_lorenzians(self):
        plt.figure()
        plt.plot(self.scan_freqs, self.interpolated_spectrum)
        for i in range(len(self.peaks_per_mode)):
            plt.plot(self.scan_freqs[self.peaks_per_mode[i]], self.interpolated_spectrum[self.peaks_per_mode[i]], 'o')
        for i in range(len(self.fit_res)):
            plt.plot(self.scan_freqs[
                     (self.peaks[i] - int(self.peaks_width[0][i])):(self.peaks[i] + int(self.peaks_width[0][i]))],
                     self.Lorenzian(self.scan_freqs[(self.peaks[i] - int(self.peaks_width[0][i])):(
                                 self.peaks[i] + int(self.peaks_width[0][i]))], *self.fit_res[i]), 'r-')

    # needed to be class method so it can be called without generating an instance
    @classmethod
    def smooth_spectrum(self,decimation,spectrum,wavelengths):
        '''
        smooth the spectrum transmission with decimation and interpulation.
        :return:
        '''
        decimated_total_spectrum = spectrum[0:-1:decimation]
        decimated_scanned_wavelengths= wavelengths[0:-1:decimation]

        cs = CubicSpline(decimated_scanned_wavelengths, decimated_total_spectrum)
        interpolated_spectrum = cs(wavelengths)
        return interpolated_spectrum

    # needed to be class method so it can be called without generating an instance
    @classmethod
    def find_peaks_in_spectrum(self,prominence,height,distance,rel_height,spectrum):
        '''
        find peaks in transmission spectrum
        :param prominence: The prominence of a peak measures how much a peak stands out from the surrounding
                           baseline of the signal and is defined as the vertical distance between the peak and its lowest contour line.
        :param height: Required height of peaks. Either a number, None, an array matching x or a 2-element sequence of the former.
                       The first element is always interpreted as the minimal and the second, if supplied, as the maximal required height.
        :param prominence: The prominence of a peak measures how much a peak stands out from the surrounding
                           baseline of the signal and is defined as the vertical distance between the peak and its lowest contour line.:
        :param distance: Required minimal horizontal distance (>= 1) in samples between neighbouring peaks.
                            Smaller peaks are removed first until the condition is fulfilled for all remaining peaks.
        :param rel_height:Used for calculation of the peaks width, thus it is only used if width is given.
                            See argument rel_height in peak_widths for a full description of its effects.
        :return: peaks_width - list  = [The widths for each peak in samples, The height of the contour lines at which
                                        the widths where evaluated
                                        (multiplied by -1 to get the real heights),Interpolated positions of left and right
                                        intersection points of a horizontal line at the respective evaluation height]
        :return: peaks - Indices of peaks in x that satisfy all given conditions.
        :return: peaks properties - A dictionary containing properties of the returned peaks which were calculated as
                                    intermediate results during evaluation of the specified conditions.
                                    ‘prominences’, ‘right_bases’, ‘left_bases’
                                    If prominence is given, these keys are accessible. See peak_prominences for a description of their content.
        '''
        peaks, peaks_properties = find_peaks(-spectrum, prominence=prominence,height=height,distance=distance)
        peaks_width = peak_widths(-spectrum, peaks, rel_height=rel_height)
        peaks_width =(peaks_width[0],-peaks_width[1],peaks_width[2],peaks_width[3])
        return peaks_width, peaks, peaks_properties

    def fit_lorenzians(self):
        #the frequencies of the scan obtained from wavelengths
        fit_res = []
        for i in range(len(self.peaks)):
            # initial guess
            kappa_guess = self.scan_freqs[self.peaks_width[2][i].astype(int)]-self.scan_freqs[self.peaks_width[3][i].astype(int)] # a guess for the aprrox width of the lorenzian [THz]
            x_dc_guess = self.scan_freqs[self.peaks[i]] # a guess for the central frequency [THz]
            y_dc_guess = self.peaks_properties["prominences"][i]+self.interpolated_spectrum[self.peaks[i]]
            initial_guess = np.array([kappa_guess/2, kappa_guess/2, x_dc_guess,y_dc_guess,y_dc_guess,0])

            x_data = self.scan_freqs[(self.peaks[i] - int(self.peaks_width[0][i])):(self.peaks[i] + int(self.peaks_width[0][i]))]
            y_data =  self.interpolated_spectrum[(self.peaks[i] - int(self.peaks_width[0][i])):(self.peaks[i] + int(self.peaks_width[0][i]))]
            popt, pcov = curve_fit(self.Lorenzian, x_data,
                                   y_data,
                               bounds=([0,0,0,y_dc_guess*0.9,0,0], [1e3, 1e3,1e5, y_dc_guess*1.1,1e8,1e-6]), p0=initial_guess)
            fit_res.append(popt)
        return fit_res

    def calc_effective_kappa(self):
        effective_kappa= []
        for i in range(len(self.fit_res)):
            effective_kappa.append(np.sqrt(self.fit_res[i][0] ** 2 + self.fit_res[i][1] ** 2))
        return effective_kappa
    def get_scan_freqs(self):
        '''

        :return: freqs - in Thz
        '''
        freqs = (C_light* 1e-12) / (self.scan_wavelengths * 1e-9)
        return freqs

    def Lorenzian(self,x, kex, ki, x_dc, y_dc,amp,h):
        return (y_dc - abs(amp * kex * (1j * (x - x_dc) + (kex + ki)) / (h ** 2 + (1j * (x - x_dc) + (kex + ki)) ** 2)) ** 2)

if __name__ == "__main__":
    o=AnalyzeSpectrum(decimation=10)