import os
import sys
import ipdb
import numpy as np

import clr

from time import sleep
from clr import System
from System.Text import StringBuilder
from System import Int32
from System.Reflection import Assembly
sys.path.append('C:\\Program Files\\New Focus\\New Focus Tunable Laser Application\\')
clr.AddReference('BasicInstrumentsControl\\Laser\\UsbDllWrap')
import Newport

class LaserControl():
    def __init__(self):
        self.tlb = Newport.USBComm.USB()
        self.answer = clr.System.Text.StringBuilder(64)

        self.ProductID = 4106
        self.DeviceKey = '6700 SN95227'

        self.tlb_open()

        #self.tlb_query('*RST')  # Performs a soft reset of the instrument.
        self.tlb_query('*IDN?')

    def __del__(self):
        self.tlb_close()

    def tlb_open(self):
        self.tlb.OpenDevices(self.ProductID, True)

    def tlb_close(self):
        self.tlb.CloseDevices()

    def tlb_query(self,msg):
        self.answer.Clear()
        self.tlb.Query(self.DeviceKey, msg, self.answer)
        return self.answer.ToString()

    def tlb_set_wavelength(self,wavelength):
        # wavelength in nm
        self.tlb_query('SOURce:WAVElength {}'.format(wavelength))
        self.tlb_query('OUTPut:TRACK 1')
        lambda_current = self.tlb_query('SOURCE:WAVELENGTH?')
        print('λ_current = {} nm'.format(lambda_current))
        return lambda_current
if __name__=='__main__':
    o=LaserControl()
    o.tlb_set_wavelength(778.58)