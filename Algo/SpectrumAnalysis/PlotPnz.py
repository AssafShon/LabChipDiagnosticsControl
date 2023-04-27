import os
import numpy as np
import matplotlib.pyplot as plt

'''
This file plot data from PNZ files
'''


class PlotPnz:
    def load(self):
        print("what is the full path of the waveguide?")
        saved_file_root = input()

        pnz_files_in_folder = [file for file in os.listdir(saved_file_root) if file.endswith('.npz')]
        if len(pnz_files_in_folder)==1:
            load_filename = pnz_files_in_folder[0]
        else:
            print('Choose the file number (0,1,2..): ')
            for i in pnz_files_in_folder:
                print(str(pnz_files_in_folder.index(i))+'. '+i)
            num = int(input())
            load_filename = pnz_files_in_folder[num]

        np_root = os.path.join(saved_file_root, load_filename)

        # second file
        second_file_root = r'C:\Users\Lab2\qs-labs\R&D - Lab\Chip Tester\Spectrum_transmission\Statstic\GDSI_T401\chip18\W1-11\TE scan'
        second_file_name = '20230427-105423Test.npz'
        np_second_root = os.path.join(second_file_root, second_file_name)
        second_data = np.load(np_second_root)
        self.second_spectrum = second_data['spectrum']
        self.second_wavelengths = second_data['wavelengths']


        data = np.load(np_root)
        self.total_spectrum = data['spectrum']
        self.scan_wavelengths = data['wavelengths']
        self.cosy = data['cosy_spectrum']
        self.cosy_wavelengths = data['cosy_wavelengths']

        self.plot()

    def plot(self):
        plt.figure()
        plt.plot(self.scan_wavelengths, self.total_spectrum, 'orange')
        plt.plot(self.cosy_wavelengths, self.cosy, 'b')
        plt.show(block=False)
        plt.pause(0.1)


if __name__ == "__main__":
    plot = PlotPnz()
    plot.load()
