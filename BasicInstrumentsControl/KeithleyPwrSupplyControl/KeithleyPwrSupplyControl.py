
import pyvisa
import time
import numpy as np
from Utility_functions import plot_figure
import matplotlib.pyplot as plt


class KeithleyPwrSupplyControl():
    def __init__(self,channel=1, current_lim=10e-3):
        self.rm = pyvisa.ResourceManager()
        self.keithley = self.KEI2231_Connect(rsrcString='ASRL4::INSTR',getIdStr=1, timeout=20000, doRst=1)
        self.channel = channel
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
        self.SelectChannel(channel=self.channel)
        self.SetCurrent(self.current_lim / 1e3)  # set the current in A
        self.OutputState(1)

        self.voltages_scan = np.arange(0, self.voltage_lim, self.voltage_lim / num_of_points)
        self.currents_scan = np.zeros(len(self.voltages_scan))
        for index, V in enumerate(self.voltages_scan):
            self.SetVoltage(V)
            self.currents_scan[index] = self.getCurrentOutput()
        self.OutputState(0)

    def estimate_avg_resistance(self, current_setpoints=[2, 4, 6],high_volts=30):
        ''''
        estimates average resistance by measuring the voltage for 3 current points and
        taking the average.
        :param current_setpoints: a list of currentsto take average on  [mA]
        '''
        self.SelectChannel(channel=self.channel)
        self.SetVoltage(high_volts)
        self.OutputState(1)
        voltage_setpoints = np.zeros(len(current_setpoints))
        for index, I in enumerate(current_setpoints):
            # to protect against high currents
            if I > self.current_lim:
                raise Exception(I + ' [mA] is higher then ' + self.current_lim
                                + ' [mA] the current limit')
            self.SetCurrent(I=(I / 1e3))  # divide by 1e3 to get [A]
            voltage_setpoints[index] = self.getVoltageOutput()
        avg_resistance = np.mean(voltage_setpoints / np.asarray(current_setpoints)) # [Ohm]
        self.OutputState(0)
        return avg_resistance

    def KEI2231_Connect(self,rsrcString, getIdStr, timeout, doRst):
        my_PS = self.rm.open_resource(rsrcString, baud_rate = 9600, data_bits = 8)	#opens desired resource and sets it variable my_instrument
        my_PS.write_termination = '\n'
        my_PS.read_termination = '\n'
        my_PS.send_end = True
        my_PS.StopBits = 1
        # my_PS.flow_control =      # only available in PyVisa 1.9
        #my_PS.baud_rate = 9600
        if getIdStr == 1:
            print(my_PS.query("*IDN?"))
            #time.sleep(0.1)
        my_PS.write('SYST:REM')
        #print(my_PS.timeout)
        my_PS.timeout = timeout
        #print(my_PS.timeout)
        if doRst == 1:
            my_PS.write('*RST')
            #time.sleep(0.1)
        return my_PS

    def Disconnect(self):
        self.keithley.write('SYST:LOC')
        self.keithley.close()
        return

    def SelectChannel(self,channel):
        self.keithley.write("INST:NSEL %d" % channel)
        #time.sleep(0.25)
        return

    def SetVoltage(self,V):
        '''

        :param V: voltage in volts
        :return:
        '''
        self.keithley.write("VOLT %f" % V)
        time.sleep(0.8)
        return

    def SetCurrent(self,I):
        '''

        :param I: the current in [A]
        :return:
        '''
        self.keithley.write("CURR %f" % I)
        time.sleep(0.8)
        return

    def getVoltageOutput(self):
        V = self.keithley.query("MEAS:VOLT?")
        return float(V)

    def getCurrentOutput(self):
        I = self.keithley.query("MEAS:CURR?")
        return float(I)

    def OutputState(self,State):
        if State == 0:
            self.keithley.write("OUTP 0")
            #time.sleep(0.25)
            #my_PS.write("OUT:ENAB 0")
        else:
            self.keithley.write("OUTP 1")
            #time.sleep(0.25)
            #my_PS.write("OUT:ENAB 1")
        #time.sleep(0.25)
        return

    def KEI2231_Send(self,sndBuffer):
        self.keithley.write(sndBuffer)
        return

    def KEI2231_Query(self,sndBuffer):
        return self.keithley.query(sndBuffer)


if __name__ == "__main__":
    o = KeithleyPwrSupplyControl()
    o.get_voltage_to_current()
    plt.show()
    o.Disconnect()

#
#
# #================================================================================
# #    MAIN CODE GOES HERE
# #================================================================================
# t1 = time.time()    # Capture start time....
# rm = pyvisa.ResourceManager() # Opens the resource manager and sets it to variable rm
# print(rm.list_resources())
# my_PS = KEI2231_Connect('ASRL4::INSTR', 1, 20000, 1)
#
# SelectChannel(1)
# SetVoltage(1.0)
# SetCurrent(1.0)
# OutputState(1)
#
# time.sleep(0.25)
#
#
# OutputState(0)
#
# Disconnect()
#
# rm.close
#
# t2 = time.time() # Capture stop time...
# print("{0:.3f} s".format((t2-t1)))
