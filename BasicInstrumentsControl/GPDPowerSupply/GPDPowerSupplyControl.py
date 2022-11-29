import gpd3303s


class PowerSupply():
    def __init__(self):
        gpd = gpd3303s.GPD3303S()
        gpd.open('COM3')  # Open the device

        v = gpd.getVoltageOutput(1)  # read the actual voltage

    def set_current(self,current,current_lim = 10):
        '''
        set current to the power supply
        :param current: current in [mA]
        :param current_lim: limits the current from the power supply[mA] (Should not exceed 10mA)
        :return:
        '''
        if current>10:
            raise 'current to power supply is too high!'
        # gpd.setVoltage(1, 1.234)  # Set the voltage of channel 1 to 1.234 (V)
        # gpd.enableOutput(True)  # Output ON