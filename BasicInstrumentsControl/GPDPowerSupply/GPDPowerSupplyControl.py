import gpd3303s
gpd = gpd3303s.GPD3303S()
gpd.open('COM3') # Open the device
gpd.setVoltage(1, 1.234) # Set the voltage of channel 1 to 1.234 (V)
gpd.enableOutput(True) # Output ON
v = gpd.getVoltageOutput(1) # read the actual voltage