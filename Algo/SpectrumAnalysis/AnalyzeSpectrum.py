from scipy.interpolate import CubicSpline
from scipy.signal import peak_widths, find_peaks
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from collections import Counter
from TransmissionSpectrum import TransmissionSpectrum
import os
import time
import math
import csv
from Utility_functions import bcolors


#parameters
C_light = 2.99792458e8

class AnalyzeSpectrum(TransmissionSpectrum):

    # initializing with specific root
    #def __init__(self, run_experiment, prominence=0.1, height=None, distance=None, rel_height=0.5,
                    #saved_file_root = r'C:\Users\Lab2\qs-labs\R&D - Lab\Chip Tester\Spectrum_transmission\assaf_dev',
                    #saved_filename = 'analysis_results', fsr=1.3, num_of_rings=4, init_frequency=384, diff_between_groups = 0.03):

    # default initializing
    def __init__(self, run_experiment, saved_file_root=None, prominence=0.1, height=None, distance=None, rel_height=0.5,
                 saved_filename='analysis_results', fsr=1.3, num_of_rings=4, init_frequency=384,
                 diff_between_groups=0.03):
        '''
        :param prominence: The prominence of a peak measures how much a peak stands out from the surrounding [normalized from 0 to 1]
                           baseline of the signal and is defined as the vertical distance between the peak and its lowest contour line.:
        :param height: Required height of peaks. Either a number, None, an array matching x or a 2-element squence of thee former.
                       The first element is always interpreted as the minimal and the second, if supplied, as the maximal required height.
        :param distance: Required minimal horizontal distance (>= 1) in samples between neighbouring peaks. Smaller peaks are removed first until the condition is fulfilled for all remaining peaks.
        :param rel_height:Used for calculation of the peaks width, thus it is only used if width is given. See argument rel_height in peak_widths for a full description of its effects.
        :param run_experiment: bool between running an experiment via Analyze spectrum or just analyzing loaded data.
        '''
        # :param max_diff_between_widths_coeff:the maximum difference between widths of modes to be considered as a group of same mode
        if run_experiment == 'True' or run_experiment == '1' or run_experiment == 'true':
            if saved_file_root == None:
                print("what is the full path of the waveguide?")
                saved_file_root = input()
            super().__init__()

            # notice - the decimation for scanning is inside TransmissionSpectrum.py
            self.get_wide_spectrum(parmeters_by_console=True)

            # this decimation is only for this plotting and saving of the scans data
            decimation_in_samples_for_scan = 10
            self.plot_transmission_spectrum(self.total_spectrum, decimation=decimation_in_samples_for_scan)
            print('For canceling the detector\'s noise, turn off the laser. What\'s the noise value? [mV]')
            detector_noise = input()
            np_root =self.save_figure_and_data(saved_file_root,
                                   self.total_spectrum,decimation_in_samples_for_scan,'', detector_noise)
            self.Pico.__del__()
            self.Laser.__del__()
        else:
            if saved_file_root == None:
                print("what is the full path of your npz file?")
                saved_file_root = input()

            pnz_files_in_folder = [file for file in os.listdir(saved_file_root) if file.endswith('.npz')]
            load_filename = pnz_files_in_folder[0]

            # saved_file_root = r'C:\Users\Lab2\qs-labs\R&D - Lab\Chip Tester\Spectrum_transmission\Statstic\07E0\chip2\W1-09'
            # load_filename = r'20230110-102329Test.npz'

            np_root = os.path.join(saved_file_root, load_filename)
            #np_root = os.path.join(self.transmission_directory_path, load_filename)

        # load saved data
        data = np.load(np_root)
        self.total_spectrum = data['spectrum']
        # compensation for the photodiode minimum value
        self.total_spectrum = self.total_spectrum - 12
        self.scan_wavelengths = data['wavelengths']

        # convert from nm to THz
        self.scan_freqs = self.get_scan_freqs(self.scan_wavelengths)


        #smooth spectrum and normalize it
        decimation = self.invert_decimatiom_from_freq_to_samples()    # this decimation is for the Analysis
        [self.interpolated_spectrum,self.interpolated_spectrum_unNorm] = self.smooth_and_normalize_spectrum(decimation, spectrum =self.total_spectrum, wavelengths=self.scan_wavelengths)

        # find peaks and divide to different modes
        self.peaks_width,self.peaks,self.peaks_properties = self.find_peaks_in_spectrum(prominence,height,distance,rel_height,spectrum=self.interpolated_spectrum)
        self.peaks_width_in_Thz_negative = [self.scan_freqs[(self.peaks[i] + int(self.peaks_width[0][i]/2))] -
                                   self.scan_freqs[(self.peaks[i] - int(self.peaks_width[0][i]/2))] for i in range(len(self.peaks))]
        self.peaks_width_in_Thz = [abs(value) for value in self.peaks_width_in_Thz_negative]

        #division_width_between_modes = self.division_width_choose()
        division_width_between_modes = 6/1000

        #divide different modes
        self.fundamental_mode,self.high_mode =self.divide_to_different_modes(peaks=self.peaks,division_width_between_modes = division_width_between_modes,modes_width =self.peaks_width_in_Thz )
        [self.peaks_fundamental_mode, self.peaks_high_mode] = [self.fundamental_mode[1],self.high_mode[1]]
        self.peaks_fund_mode_ind = self.fundamental_mode[0]
        self.peaks_per_mode = [self.peaks_fundamental_mode, self.peaks_high_mode]

        #plot figure with peaks
        self.plot_peaks(scan_freqs=self.scan_freqs,interpolated_spectrum=self.interpolated_spectrum,
                        peaks_per_mode=self.peaks_per_mode, peaks_properties=self.peaks_properties)

        #fit lorenzians
        self.width_increase = 1 # multipy this factor to the width of peak for the width of fit
        [self.fit_res,self.fit_cov_params] = self.fit_lorenzians()


        self.effective_kappa_all_resonances = self.calc_effective_kappa_and_h()

        #plot lorenzians
        self.plot_lorenzians()
        plt.show()

        # classify peaks to different rings
        # self.classify_peaks(fsr, num_of_rings, init_frequency,diff_between_groups)

        #get parameters and save them
        self.get_analysis_spectrum_parameters()
        if run_experiment == 'True' or run_experiment == '1' or run_experiment == 'ture':
            self.save_analyzed_data(dist_root=self.transmission_directory_path, filename=saved_filename
                                    , analysis_spectrum_parameters=self.analysis_spectrum_parameters,
                                    spectrum_data=[self.interpolated_spectrum,self.scan_freqs,self.fit_res,self.peaks,
                                    self.peaks_width],figure=self.lorenzian_fig)
        else:
            self.save_analyzed_data(dist_root=saved_file_root, filename=saved_filename
                                    , analysis_spectrum_parameters=self.analysis_spectrum_parameters,
                                    spectrum_data=[self.interpolated_spectrum, self.scan_freqs, self.fit_res,
                                                   self.peaks, self.peaks_width],figure=self.lorenzian_fig)


    @classmethod
    def save_analyzed_data(self,dist_root,filename,analysis_spectrum_parameters, spectrum_data,figure, width_fig=None):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        # create directory
        analysis_path = dist_root+r'\analysis'
        if not os.path.isdir(analysis_path):
            os.mkdir(analysis_path)
        # create directory
        analysis_path_with_time = os.path.join(analysis_path,timestr)
        os.mkdir(analysis_path_with_time)
        # save figure
        figure.savefig(os.path.join(analysis_path_with_time,timestr+filename+'.png'))
        if width_fig!=None:
            width_fig.savefig(os.path.join(analysis_path_with_time, timestr + filename + 'colored_widths.png'))
        # save data as csv
        if not analysis_spectrum_parameters is None:
            prameters_csv = os.path.join(analysis_path_with_time,'parameters_'+timestr+filename+'.csv')
            with open(prameters_csv, 'w') as f:
                for key in analysis_spectrum_parameters.keys():
                    f.write("%s,%s\n"%(key,analysis_spectrum_parameters[key]))
            # save python data
            np.savez(os.path.join(analysis_path_with_time,'parameters_'+timestr+filename+'.npz'),
                     parameters=analysis_spectrum_parameters)

        # save figure data in python
        np.savez(os.path.join(analysis_path_with_time, 'spectrum_data_'+timestr + filename + '.npz'),
                 spectrum=spectrum_data)

    def classify_peaks(self,fsr, num_of_rings, init_frequency,diff_between_groups):
        '''
        classify peaks to their fsr and ring number
        :param fsr - the distance between peaks
        :param num of rings - number of rings
        :param init_frequency- the frequency of a peak of a first ring
        :return:
        '''
        self.peak_groups = self.divide_into_peak_groups(init_frequency=init_frequency,fsr=fsr)
        classified_peaks = self.relate_pk_to_ring(init_frequency=init_frequency,num_of_rings=num_of_rings,fsr=fsr,diff_between_groups=diff_between_groups)
        return classified_peaks

    def divide_into_peak_groups(self,init_frequency,fsr):
        '''
        divides a group of oeaks intogroups of different resonances
        :param fsr_dist - the distance between peaks
        :param num of rings - number of rings
        :param init_frequency- the frequency of a peak of a first ring
        :return:full_pk_group - a group of peaks in the length of num_of_peaks
        :return:single_fsr_peaks - a group of peaks divided into subgroups of different fsrs
        '''
        peak_groups = []
        fsr_init_freq = init_frequency
        if self.peaks_fundamental_mode == []:   #prevent an error when there is no fundamental mode peaks
            return peak_groups
        while fsr_init_freq<self.scan_freqs[self.peaks_fundamental_mode[0]] :
            peak_groups.append([peak for peak in self.peaks_fundamental_mode if fsr_init_freq<self.scan_freqs[peak]<(fsr_init_freq+fsr)])
            fsr_init_freq += fsr
        return peak_groups

    def relate_pk_to_ring(self,init_frequency,fsr,num_of_rings,diff_between_groups):
        '''

        relates peaks of fundamental mode to their ring.
        peak_group_relative_dist - the peaks divided to fsr and each group includes the distance from initial frequency of fsr
        diff_between_groups - the maximal distance in THz between peaks from different groups related to the same ring
        :return: classified_peaks - a group of cpouples: (ring,peak)

        '''
        full_pk_group = []
        classified_peaks = []
        for i in range(len(self.peak_groups)):
            classified_peaks.append([])  # In each iteration, add an empty list to the main list

        # generate groups of peaks all relatively distanced from the initial frequency
        peak_group_relative_dist = [[peak -(init_frequency+fsr*fsr_num) for peak in peak_group]
                                    for fsr_num, peak_group in enumerate(self.peak_groups)]

        # find a group with num_of_rings peaks to be a reference for order peaks
        for peak_group in range(len(peak_group_relative_dist)):
            if len(peak_group_relative_dist[peak_group]) == num_of_rings:
                full_pk_group = peak_group_relative_dist[peak_group]
                break
        if full_pk_group == []:
            print(bcolors.WARNING + "Warning: There is no FSR with peaks as the number of rings"
                                    ", can not classify peak to different rings" + bcolors.ENDC)
            return classified_peaks

        #  generate group of couples: (ring,peak) by comparing to full_pk_group
        for peak_group in range(len(peak_group_relative_dist)):
            for i in range(num_of_rings):
                numpy_pk_group = np.asarray(peak_group_relative_dist[peak_group])
                peak = np.where(numpy_pk_group-full_pk_group[i]<diff_between_groups)
                classified_peaks[peak_group].append((peak_group_relative_dist[peak_group][peak[0][0]],i))
        return classified_peaks

    def division_width_choose(self):
        # plot: 1. the spectrum
        # 2. a histogram of the peaks widths
        # 3. list of the widths
        # than asking the user for division width between the modes
        self.plot_peaks(scan_freqs=self.scan_freqs, interpolated_spectrum=self.interpolated_spectrum,
                        peaks_per_mode=[self.peaks])  # plot the spectrum before division
        peaks_width_in_Ghz = [int(i * 1000) for i in self.peaks_width_in_Thz]

        plt.figure()
        plt.bar(list(range(1, 21)), np.histogram(peaks_width_in_Ghz, int(np.ptp(peaks_width_in_Ghz)))[0][:20])
        #plt.plot(np.histogram(peaks_width_in_Ghz,int(np.ptp(peaks_width_in_Ghz)/20))[1][:-1],np.histogram(peaks_width_in_Ghz,int(np.ptp(peaks_width_in_Ghz)/10))[0])
        plt.ylabel('Number of peaks')
        plt.xlabel('Width [Ghz]')
        plt.pause(0.1)
        plt.show(block=False)

        widths_dictionary = Counter(peaks_width_in_Ghz)
        sorted_keys = sorted(list(widths_dictionary.keys()))
        sorted_widths_dictionary = {i: widths_dictionary[i] for i in sorted_keys}
        print("list of peaks widths-  width [GHz] : number of peaks\n"+str(sorted_widths_dictionary))
        print("choose width value for division between modes:")
        division_value = int(input())
        plt.close('all')
        return division_value / 1000

    @classmethod
    def divide_to_different_modes(self,peaks,modes_width, division_width_between_modes):  # max_diff_between_widths_coeff=0.1):
        '''
        divides peaks into different modes depending on their width
        :param diff_condition_between_modes_width - defines the difference in mode width to be considered as the same mode
        :return:
        '''
        # cl = cluster.HierarchicalClustering(modes_width, lambda x, y: abs(x - y))
        # self.peaks_width_per_mode = cl.getlevel(max_diff_between_widths_coeff*np.mean(modes_width)) fundamental

        widths_fundamental_mode = [a for a in modes_width if a<division_width_between_modes]
        peaks_fund_mode_ind = [j for j, x in enumerate(modes_width) if x in widths_fundamental_mode]
        peaks_fundamental_mode = [peaks[k] for k in peaks_fund_mode_ind]
        fundamental_mode = [widths_fundamental_mode, peaks_fundamental_mode, peaks_fund_mode_ind]

        widths_high_mode = [a for a in modes_width if a>division_width_between_modes]
        peaks_high_mode_ind = [j for j, x in enumerate(modes_width) if x in widths_high_mode]
        peaks_high_mode = [peaks[k] for k in peaks_high_mode_ind]
        high_mode = [widths_high_mode, peaks_high_mode, peaks_high_mode_ind]

        return fundamental_mode,high_mode

    @classmethod
    def plot_peaks(self,scan_freqs,interpolated_spectrum,peaks_per_mode,peaks_properties):
        peaks_fig = plt.figure()
        plt.plot(scan_freqs, interpolated_spectrum)
        for i in range(len(peaks_per_mode)):
            plt.plot(scan_freqs[peaks_per_mode[i]], interpolated_spectrum[peaks_per_mode[i]], 'o')
            # this line supposed to print the peak widths as the python function found them
            #plt.hlines(y=interpolated_spectrum[peaks_per_mode[i]], xmin=scan_freqs[peaks_properties["left_bases"]],
            #        xmax=scan_freqs[peaks_properties["right_bases"]], color="C1")
        plt.pause(0.1)
        plt.show(block=False)
        return peaks_fig

    def plot_lorenzians(self):
        self.lorenzian_fig = plt.figure()
        plt.plot(self.scan_freqs, self.interpolated_spectrum)
        for i in range(len(self.peaks_per_mode)):
            plt.plot(self.scan_freqs[self.peaks_per_mode[i]], self.interpolated_spectrum[self.peaks_per_mode[i]], 'o')
        for i in range(len(self.fit_res)):
            if len(self.interpolated_spectrum[(self.peaks[i] - int(self.peaks_width[0][i]*self.width_increase )):(self.peaks[i] + int(self.peaks_width[0][i]*self.width_increase))]) == 0:
                fix_normalitation = 1
            else:
                fix_normalitation = np.max(self.interpolated_spectrum[(self.peaks[i] - int(self.peaks_width[0][i]*self.width_increase )):(self.peaks[i] + int(self.peaks_width[0][i]*self.width_increase))])
            plt.plot(self.scan_freqs[
                     (self.peaks[i] - int(self.peaks_width[0][i]*self.width_increase )):(self.peaks[i] + int(self.peaks_width[0][i]*self.width_increase ))],
                     fix_normalitation*self.Lorenzian(self.scan_freqs[(self.peaks[i] - int(self.peaks_width[0][i]*self.width_increase )):(
                             self.peaks[i] + int(self.peaks_width[0][i]*self.width_increase))], *self.fit_res[i]), 'r-')

    @classmethod
    def plot_peaks_colored_by_width(self,ax,peaks_freqs,Y,colors):
        '''
        plot peaks colored by width
        '''
        ax.scatter(peaks_freqs,Y, c=colors)


    # needed to be class method so it can be called without generating an instance
    @classmethod
    def smooth_and_normalize_spectrum(self, decimation, spectrum, wavelengths):
        '''
        smooth the spectrum transmission with decimation and interpulation.
        :return:
        '''
        decimated_total_spectrum = spectrum[0:-1:decimation]
        decimated_scanned_wavelengths = wavelengths[0:-1:decimation]

        cs = CubicSpline(decimated_scanned_wavelengths, decimated_total_spectrum)
        #wavelengths = np.linspace(min(wavelengths),min(wavelengths),len(wavelengths))

        interpolated_spectrum = cs(wavelengths)
        norm_interpolated_spectrum = interpolated_spectrum/max(interpolated_spectrum)
        return norm_interpolated_spectrum,interpolated_spectrum

    # needed to be class method so it can be called without generating an instance
    @classmethod
    def find_peaks_in_spectrum(self,prominence,height,distance,rel_height,spectrum):
        '''
        find peaks in transmission spectrum
        :param prominence: The prominence of a peak measures how much a peak stands out from the surrounding
                           baseline of the signal and is defined as the vertical distance between the peak and its lowest contour line.
        :param height: Required height of peaks. Either a number, None, an array matching x or a 2-element sequence of the former.
                       The first element is always interpreted as the minimal and the second, if supplied, as the maximal required height.
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
        peaks_width = (peaks_width[0],-peaks_width[1],peaks_width[2],peaks_width[3])
        return peaks_width, peaks, peaks_properties

    def fit_lorenzians(self):
        '''

        :return: fit_quality - the standard deviation errors on the parameters
        '''
        #the frequencies of the scan obtained from wavelengths
        fit_parameters = []
        fit_quality = []
        for i in range(len(self.peaks)):
            # initial guess
            kappa_guess = abs(self.scan_freqs[self.peaks_width[3][i].astype(int)]-self.scan_freqs[self.peaks_width[2][i].astype(int)]) # a guess for the aprrox width of the lorenzian [THz]
            x_dc_guess = self.scan_freqs[self.peaks[i]] # a guess for the central frequency [THz]
            y_dc_guess = self.peaks_properties["prominences"][i]+self.interpolated_spectrum[self.peaks[i]]
            amp_guess = y_dc_guess-self.interpolated_spectrum[self.peaks[i]]
            initial_guess = np.array([kappa_guess/2, kappa_guess/2, x_dc_guess,y_dc_guess,amp_guess,0])

            # self.width_increase = 1.7
            self.width_increase = self.curve_fit_for_diff_base_widths(kappa_guess, x_dc_guess, y_dc_guess, amp_guess, initial_guess, i)

            # make sure the start ind.2ex of x_data and y_data isn't lower than 0
            if self.peaks[i] < int(self.peaks_width[0][i] * self.width_increase):
                start_index = 0;
            else:
                start_index = (self.peaks[i] - int(self.peaks_width[0][i] * self.width_increase))

            x_data = self.scan_freqs[start_index:(self.peaks[i] + int(self.peaks_width[0][i] * self.width_increase))]
            y_data = self.interpolated_spectrum_unNorm[
                     start_index:(self.peaks[i] + int(self.peaks_width[0][i] * self.width_increase))]

            try:
                popt, pcov = curve_fit(self.Lorenzian, x_data,
                                       y_data / max(y_data),
                                       bounds=([0, 0, 0, y_dc_guess * 0.5, amp_guess / 3, 0],
                                               [1e3, 1e3, 1e3, 1, amp_guess * 3, 1]), p0=initial_guess)

                fit_parameters.append(popt)
                fit_quality.append(np.sqrt(np.diag(pcov)))

            except Exception:
                print(bcolors.WARNING + "Warning: peak number " + str(
                    i) + " could not be fitted to lorenzian" + bcolors.ENDC)
                # prevent error in case curve_fit didn't succed
                popt = np.zeros(6)
                pcov = np.zeros((6,6))
                fit_parameters.append(0 * popt)
                fit_quality.append(np.sqrt(np.diag(0 * pcov)))
        return [fit_parameters, fit_quality]

    def curve_fit_for_diff_base_widths(self,kappa_guess,x_dc_guess,y_dc_guess,amp_guess,initial_guess, i):
        '''
        run curve_fit function with set of different widths of x_data and y_data: [4, 3.8, 3.6,... 2]
        and choose automatically the best fit
        the best fit = ???
        '''

        width_increase = 1
        cov_values = {}
        stop = 0

        # make sure the start ind.2ex of x_data and y_data isn't lower than 0
        if self.peaks[i] < int(self.peaks_width[0][i] * width_increase):
            start_index = 0
        else:
            start_index = (self.peaks[i] - int(self.peaks_width[0][i] * width_increase))

        # This params are for comparing between the different iterations
        x_data_peak = self.scan_freqs[start_index:(self.peaks[i] + int(self.peaks_width[0][i] * width_increase))]
        y_data_peak = self.interpolated_spectrum_unNorm[
                 start_index:(self.peaks[i] + int(self.peaks_width[0][i] * width_increase))]

        print('Peak number '+str(i)+', in frequency '+str(self.scan_freqs[self.peaks[i]])[:7])
        if i == 1 or i == 2 or i == 12:
            width_increase = 0.2
        while width_increase <= 4 and stop == 0:

            # make sure the start ind.2ex of x_data and y_data isn't lower than 0
            if self.peaks[i] < int(self.peaks_width[0][i] * width_increase):
                start_index = 0
            else:
                start_index = (self.peaks[i] - int(self.peaks_width[0][i] * width_increase))

            # This params goes into curve_fit function
            x_data = self.scan_freqs[start_index:(self.peaks[i] + int(self.peaks_width[0][i] * width_increase))]
            y_data = self.interpolated_spectrum_unNorm[
                     start_index:(self.peaks[i] + int(self.peaks_width[0][i] * width_increase))]

            try:
                popt, pcov = curve_fit(self.Lorenzian, x_data,
                                       y_data / max(y_data),
                                       bounds=([0, 0, 0, y_dc_guess * 0.5, amp_guess / 3, 0],
                                               [1e3, 1e3, 1e3, 1, amp_guess * 3, 1]), p0=initial_guess)

                # fit RMS error
                error_vec = self.Lorenzian(x_data_peak, popt[0],popt[1],popt[2],popt[3],popt[4],popt[5])-(y_data_peak / max(y_data))
                cov_values[width_increase] = sum(error_vec ** 2)

                # plot the fit and the real data for a single peak and width_increase val
                # plt.figure()
                # plt.plot(self.Lorenzian(x_data, popt[0],popt[1],popt[2],popt[3],popt[4],popt[5]),'g')
                # plt.plot(y_data / max(y_data), 'r')
                # plt.title('Lorenzian fit for width increase '+str(width_increase))
                # plt.pause(0.1)
                # plt.show(block=False)

                # print("Is the fit ok? [1-yes, 0-no, try small width]")
                # stop = int(input())


            except Exception:
                print(bcolors.WARNING + "For width_increase = " + str(width_increase) + ", peak number " + str(
                    i) + " could not be fitted to lorenzian" + bcolors.ENDC)


            width_increase = round(width_increase+0.2, 2)
        plt.close('all')
        return min(cov_values, key=cov_values.get)

    def get_analysis_spectrum_parameters(self):
        # generates a list of resonances with all parameters
        self.analysis_spectrum_parameters = {}
        self.analysis_spectrum_parameters['mode[THz]'] = ["fundamental" if i in self.peaks_fund_mode_ind else "high"
                                               for i in self.peaks_width_in_Thz]
        self.analysis_spectrum_parameters['peak_freq[GHz]'] = [round(elem,3) for elem in  self.scan_freqs[self.peaks]]
        self.analysis_spectrum_parameters['kappa_ex[GHz]'] = [round(self.fit_res[i][0]*1e3,3) for i in range(len(self.fit_res))]
        self.analysis_spectrum_parameters['kappa_i[GHz]'] = [round(self.fit_res[i][1]*1e3,3) for i in range(len(self.fit_res))]
        self.analysis_spectrum_parameters['h'] = [round(self.fit_res[i][5]*1e3,3) for i in range(len(self.fit_res))]
        # The FWHM is the width in 0.5 peaks height, define with 'rel_height'
        self.analysis_spectrum_parameters['FWHM[GHz]'] = [round(self.peaks_width_in_Thz[i]*1e3,3) for i in range(len(self.peaks_width_in_Thz))]
        # self.analysis_spectrum_parameters['standard deviation of kappa_ex'] = [self.fit_cov_params[i][0] for i in range(len(self.fit_cov_params))]
        # self.analysis_spectrum_parameters['standard deviation of kappa_i'] = [self.fit_cov_params[i][1] for i in range(len(self.fit_cov_params))]
        # self.analysis_spectrum_parameters['standard deviation of h'] = [self.fit_cov_params[i][5] for i in range(len(self.fit_cov_params))]

    def calc_effective_kappa_and_h(self):
        # returns the geometric average of kappa i, kappa ex and h
        effective_kappa= []
        for i in range(len(self.fit_res)):
            effective_kappa.append(np.sqrt(self.fit_res[i][0] ** 2 + self.fit_res[i][1] ** 2 + self.fit_res[i][5] ** 2))
        return effective_kappa

    @classmethod
    def get_scan_freqs(self,scan_wavelengths):
        '''

        :return: freqs - in Thz
        '''
        freqs = (C_light * 1e-12) / (scan_wavelengths * 1e-9)
        #freqs = np.flip(freqs)
        return freqs

    def Lorenzian(self,x, kex, ki, x_dc, y_dc,amp,h):
        #return abs(y_dc*(1 - 2 *( kex * (1j * (x - x_dc) + (kex + ki)) / (h ** 2 + (1j * (x - x_dc) + (kex + ki)) ** 2)) ** 2))
        #return abs((1 - (2 * kex * (1j * (x - x_dc) + (kex + ki)) / (h ** 2 + (1j * (x - x_dc) + (kex + ki)) ** 2)))) ** 2
        return (1 -abs( 2 * kex * (1j * (x - x_dc) + (kex + ki)) / (h ** 2 + (1j * (x - x_dc) + (kex + ki)) ** 2)) ** 2)

    def invert_decimatiom_from_freq_to_samples(self):
        avg_freqs_diff_between_samples = abs(np.mean(np.diff(self.scan_freqs)))  #in THz
        avg_freqs_diff_between_samples = avg_freqs_diff_between_samples*1e3 #in GHz
        #print("what is the resolution, in GHz (The current resolution is %.2f GHz)?" % avg_freqs_diff_between_samples)
        #resolution_in_GHz = float(input())
        resolution_in_GHz = float(0.6)
        resolution_in_samples = int(np.ceil(resolution_in_GHz/avg_freqs_diff_between_samples))
        print("The resolution in samples is: " + str(resolution_in_samples))
        return resolution_in_samples


if __name__ == "__main__":
    print("Do you want to run a frequency scan? (True/False)")
    run_experiment = input()
    o = AnalyzeSpectrum(run_experiment=run_experiment)


