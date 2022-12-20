import gpd3303s
import numpy as np
import matplotlib.pyplot as plt
from Utility_functions import plot_figure


class PowerSupply():
    def __init__(self, Channel=1, current_lim=10):
        # connect
        self.gpd = gpd3303s.GPD3303S()
        self.gpd.open('COM5')  # Open the device

        self.channel = Channel
        self.current_lim = current_lim

    def get_voltage_to_current(self):
        '''
        creates a figure of voltage to current up to 10 mA
        :return:
        '''
        self.avg_resistance = self.estimate_avg_resistance()

        self.voltage_lim = self.current_lim * self.avg_resistance

        self.scan_voltage()

        # plot
        plot_figure(X=self.voltages_scan, Y=self.currents_scan, xlabel='Voltage[V]'
                    , ylabel='Current[A]', title='Voltage vs. Current to Heater')

    def scan_voltage(self, num_of_points=10):
        '''
        scan the voltage so the current would scan between 0- current_lim [mA].
        :param num_of_points: the number of points in scan
        :return:
        '''

        self.gpd.setCurrent(self.channel, self.current_lim / 1e3)  # set the current
        self.voltages_scan = np.arange(0, self.voltage_lim, self.voltage_lim / num_of_points)
        self.currents_scan = np.array(len(self.voltages_scan))
        for index, V in enumerate(self.voltages_scan):
            self.gpd.setVoltage(self.channel, V / 1e3)  # divide by 1e3 to get [A]
            self.currents_scan[index] = self.gpd.getCurrentOutput(self.channel)

    def estimate_avg_resistance(self, current_setpoints=[2, 4, 6]):
        ''''
        estimates average resistance bymeasuring the voltage for 3 current points and
        taking the average.
        :param current_setpoints: a list of currentsto take average on  [mA]
        '''
        self.gpd.setVoltage(self.channel, 30)
        voltage_setpoints = np.zeros(len(current_setpoints))
        for index, I in enumerate(current_setpoints):
            if I > self.current_lim:
                raise Exception(I + ' [mA] is higher then ' + self.current_lim
                                + ' [mA] the current limit')
            self.gpd.setCurrent(self.channel, I / 1e3)  # divide by 1e3 to get [A]
            voltage_setpoints[index] = self.gpd.getVoltageOutput(self.channel)
        avg_resistance = np.mean(voltage_setpoints / np.asarray(current_setpoints))
        return avg_resistance

    def set_current(self, current, current_lim=10):
        '''
        set current to the power supply
        :param current: current in [mA]
        :param current_lim: limits the current from the power supply[mA] (Should not exceed 10mA)
        :return:
        '''
        if current > 10:
            raise 'current to power supply is too high!'
        self.gpd.setVoltage(1, 1.234)  # Set the voltage of channel 1 to 1.234 (V)
        self.gpd.enableOutput(True)  # Output ON


if __name__ == "__main__":
    o = PowerSupply()
    o.gpd.enableOutput()
