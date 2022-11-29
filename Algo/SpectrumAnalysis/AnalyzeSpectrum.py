
from scipy.interpolate import CubicSpline
from scipy.signal import peak_widths, find_peaks
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#parameters
C_light= 2.99792458e8

class AnalyzeSpectrum():
    def __init__(self, decimation = 1000,
                 prominence=20, height=None, distance=None, rel_height=0.5,run_experiment=False):
        if run_experiment:
            pass
        else:
            data = np.load(
                r'C:\Users\asafs\qs-labs\R&D - Lab\Chip Tester\Spectrum_transmission\20221123-103148Transmission_spectrum.npz')
            self.total_spectrum = data['spectrum']
            self.scan_wavelengths = data['wavelengths']
        self.smooth_spectrum(decimation)
        self.peaks_width,self.peaks = self.find_peaks_in_spectrum(prominence,height,distance,rel_height)
        #plot figure with peaks
        plt.figure()
        plt.plot(self.interpulated_spectrum)
        plt.plot(self.peaks,self.interpulated_spectrum[self.peaks],'o')
        plt.hlines(*self.peaks_width[1:])

        self.lorenzians_width = self.fit_lorenzians()

        plt.figure()
        plt.plot(self.scan_freqs,self.Lorenzian(self.scan_freqs, *self.popt), 'r-')
        plt.show()

    def smooth_spectrum(self,decimation):
        '''
        smooth the spectrum transmission with decimation and interpulation.
        :return:
        '''
        self.decimated_total_spectrum = self.total_spectrum[0:-1:decimation]
        self.decimated_scanned_wavelengths= self.scan_wavelengths[0:-1:decimation]

        cs = CubicSpline(self.decimated_scanned_wavelengths, self.decimated_total_spectrum)
        self.interpulated_spectrum = cs(self.scan_wavelengths)

    def find_peaks_in_spectrum(self,prominence,height,distance,rel_height):
        '''
        find peaks in transmission spectrum
        :param prominence: The prominence of a peak measures how much a peak stands out from the surrounding
                           baseline of the signal and is defined as the vertical distance between the peak and its lowest contour line.
        :return:
        '''
        peaks, _ = find_peaks(-self.interpulated_spectrum, prominence=prominence,height=height,distance=distance)
        peaks_width = peak_widths(-self.interpulated_spectrum, peaks, rel_height=rel_height)
        peaks_width =(peaks_width[0],-peaks_width[1],peaks_width[2],peaks_width[3])
        return peaks_width,peaks

    def fit_lorenzians(self):
        #the frequencies of the scan obtained from wavelengths
        self.scan_freqs = self.get_scan_freqs()
        for i in range(len(self.peaks)):
            initial_guess = np.array([self.peaks_width[0][i]/2, self.peaks_width[0][i]/2, self.peaks[i],0])
            self.popt[i], pcov = curve_fit(self.Lorenzian, self.scan_freqs[(self.peaks[i] - int(self.peaks_width[0][i])):(self.peaks[i] + int(self.peaks_width[0][i]))],
                                   self.interpulated_spectrum[(self.peaks[i] - int(self.peaks_width[0][i])):(self.peaks[i] + int(self.peaks_width[0][i]))],
                               bounds=([0, 0, -200000,0], [1e10, 1e4, 200000,1e3]), p0=initial_guess)

    def get_scan_freqs(self):
        '''

        :return: freqs - in Mhz
        '''
        freqs = (C_light* 1e-6) / (self.scan_wavelengths * 1e-9)
        return freqs

    def Lorenzian(self,x, kex, ki, dc, h):
        return (abs(1 - 2 * kex * (1j * (x - dc) + (kex + ki)) / (h ** 2 + (1j * (x - dc) + (kex + ki)) ** 2)) ** 2)

if __name__ == "__main__":
    o=AnalyzeSpectrum(decimation=10)