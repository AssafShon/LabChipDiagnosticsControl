import os
import numpy as np
import matplotlib.pyplot as plt

'''
This file plot data from PNZ files
'''

def plot(axis_x, axis_y):
        plt.figure()
        plt.plot(axis_x, axis_y)
        plt.show(block=False)
        plt.pause(0.1)

def load():
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
    data = np.load(np_root)
    total_spectrum = data['spectrum']
    scan_wavelengths = data['wavelengths']

    plot(scan_wavelengths, total_spectrum)


if __name__ == "__main__":
    load()
