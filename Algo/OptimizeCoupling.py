from scipy.optimize import minimize
import numpy as np

from BasicInstrumentsControl.PicoControl.PicoControl import PicoControl as Pico
from BasicInstrumentsControl.PicoControl.PicoControl import PicoScopeControl as Scope
from BasicInstrumentsControl.Laser.LaserControl import LaserControl as Laser
from BasicInstrumentsControl.StepperControl.StepperControl import StepperControl as Stage

class OptimizeCoupling:
    def __init__(self,range, r0 = np.asarray([0,0,0])):
        '''

        :param r0: initial guess for the optimization process
        '''
        # connect to instruments
        self.Pico = Pico()
        self.Scope = Scope(pico=self.Pico)
        self.Laser = Laser()
        self.Stage = Stage()

        # Set safety range
        self.set_safety_range(range)

        # Minimization
        self.r0 = r0
        minimize(self.measure_transmission,self.r0)

    def measure_transmission(self,x_fiber,y_fiber,z_fiber):
        if x_fiber>self.range:
            raise "causion! the fiber is too close to the chip"
        self.Stage.Move_Stepper_To_absolute_Position_In_XYZ(x_fiber,y_fiber,z_fiber)
        self.trace = self.Scope.get_trace()
        transmission = sum(self.trace)
        return transmission

    def set_safety_range(self,range=3):
        self.range = range
