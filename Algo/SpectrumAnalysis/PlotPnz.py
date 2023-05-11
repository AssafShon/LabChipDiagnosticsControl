import os
import numpy as np
import matplotlib.pyplot as plt

'''
This file plot data from PNZ files
'''


class PlotPnz:
    def load(self):

        # first file
        saved_file_root = r'C:\Users\Lab2\qs-labs\R&D - Lab\Chip Tester\Spectrum_transmission\unnamed scans\20230420-114734'
        pnz_files_in_folder = [file for file in os.listdir(saved_file_root) if file.endswith('.npz')]
        load_filename = pnz_files_in_folder[0]
        np_root = os.path.join(saved_file_root, load_filename)

        data = np.load(np_root)
        self.total_spectrum = data['spectrum']
        self.scan_wavelengths = data['wavelengths']
        self.cosy = data['cosy_spectrum']
        self.trace_limits = data['trace_limits']

        # # second file
        # second_file_root = r'C:\Users\Lab2\qs-labs\R&D - Lab\Chip Tester\Spectrum_transmission\unnamed scans\W1-10 2^15'
        # pnz_files_in_second_folder = [file for file in os.listdir(second_file_root) if file.endswith('.npz')]
        # second_file_name = pnz_files_in_second_folder[0]
        # np_second_root = os.path.join(second_file_root, second_file_name)
        #
        # second_data = np.load(np_second_root)
        # self.second_spectrum = second_data['spectrum']
        # self.second_wavelengths = second_data['wavelengths']

        self.plot()
        #plt.plot(self.scan_wavelengths[2126:2134],self.total_spectrum[2126:2134])

    def plot(self):
        # plt.figure()
        # plt.plot(self.scan_wavelengths, self.total_spectrum, 'orange')
        # #plt.plot(self.scan_wavelengths, self.cosy, 'yellow')
        # plt.plot(self.second_wavelengths, self.second_spectrum, 'red')
        # #plt.plot(self.SigGen_spectrum, 'b')
        # plt.legend(['full spectrum', '772-775'])
        # plt.show(block=False)
        # plt.pause(0.1)
        decimation = 1
        plt.figure()
        plt.title('Transmission Spectrum X axis check')
        plt.xlabel('Wavelength[nm]')
        plt.ylabel('Voltage[mV]')
        plt.grid(True)
        #plt.plot(self.scan_wavelengths[0:-1:decimation], self.SigGen_spectrum[0:-1:decimation] / 1000, 'g')
        # plt.plot(self.scan_wavelengths[0:-1:decimation], self.SigGen_spectrum[0:-1:decimation], 'g')
        plt.plot(self.scan_wavelengths[0:-1:decimation], self.total_spectrum[0:-1:decimation], 'orange')
        self.trace_limits = [int(i) for i in self.trace_limits]
        for i in self.trace_limits:
            if i > 50 and i < (len(self.total_spectrum)-50):
                print(i)
                plt.plot(self.scan_wavelengths[i-20: i+20], self.total_spectrum[i-20: i+20], 'green')

        plt.show()





if __name__ == "__main__":
    plot = PlotPnz()
    plot.load()

# print("what is the full path of the waveguide?")
# saved_file_root = input()
#
# pnz_files_in_folder = [file for file in os.listdir(saved_file_root) if file.endswith('.npz')]
# if len(pnz_files_in_folder)==1:
#     load_filename = pnz_files_in_folder[0]
# else:
#     print('Choose the file number (0,1,2..): ')
#     for i in pnz_files_in_folder:
#         print(str(pnz_files_in_folder.index(i))+'. '+i)
#     num = int(input())
#     load_filename = pnz_files_in_folder[num]
#
# np_root = os.path.join(saved_file_root, load_filename)