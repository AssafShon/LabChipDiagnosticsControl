from TransmissionSpectrum import TransmissionSpectrum
import numpy as np
import time
import matplotlib.pyplot as plt
from BasicInstrumentsControl.PicoControl.PicoControl import PicoControl as Pico
WAIT_TIME = 0.1

class HeaterScan(TransmissionSpectrum):
    def __init__(self,resonance_wavelength=776,max_current_scan=5,num_of_points_in_scan=5):
        super().__init__()
        self.resonance_wavelength = resonance_wavelength
        self.max_current_scan = max_current_scan
        self.num_of_points_in_scan = num_of_points_in_scan

    def scan_current_to_heater(self):
        '''
        scan the current from the current supply, takes trace with pico, detects the resonance peak and plots
         the peak wavelength as function of current. Must restrict the current to 10mA.
        :return:
        '''
        self.Laser.tlb_set_wavelength(self.resonance_wavelength)

        for idx,current in enumerate(np.arange(0, self.max_current_scan,self.num_of_points_in_scan)):
            #set current to power supply
            self.PowerSupply.set_current(current)
            time.sleep(WAIT_TIME)

            #get trace from scope and detect resonance center
            self.Scope.get_trace()
            self.resonance_center[idx] = self.AnalyzeSpectrum.get_peaks()




if __name__ == "__main__":
    o = HeaterScan()
    o.scan_current_to_heater()
    plt.plot(np.arange(0, o.max_current_scan,o.num_of_points_in_scan),o.resonance_center)