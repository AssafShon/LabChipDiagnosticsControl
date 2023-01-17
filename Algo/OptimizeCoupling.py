from scipy.optimize import minimize
import numpy as np

from BasicInstrumentsControl.PicoControl.PicoControl import PicoControl as Pico
from BasicInstrumentsControl.PicoControl.PicoControl import PicoScopeControl as Scope
# from BasicInstrumentsControl.Laser.LaserControl import LaserControl as Laser
from BasicInstrumentsControl.StepperControl.StepperControl import StepperControl as Stage

class OptimizeCoupling:
    def __init__(self,x_safety_limit):
        '''

        :param r0: initial guess for the optimization process
        '''
        # connect to instruments
        self.Pico = Pico()
        self.Scope = Scope(pico=self.Pico)
        # self.Laser = Laser()
        self.Stage = Stage()

        # Set safety range
        self.set_safety_limit(x_safety_limit)


    def measure_transmission(self,fiber_location):
        x_fiber, y_fiber, z_fiber = fiber_location
        if x_fiber>self.x_safety_limit:
            raise "causion! the fiber is too close to the chip"
        self.Stage.Move_Stepper_To_absolute_Position_In_XYZ(x_fiber,y_fiber,z_fiber)
        self.trace = self.Scope.get_trace()
        transmission = np.mean(self.trace[1])
        return transmission

    def optimize_coupling(self, r0 = np.array([1.651, 2.3, 2.49]),bounds =((1.65, 1.652), (2.2, 2.4), (2.4, 2.6))):
        # Minimization
        self.r0 = r0
        self.bounds = bounds
        minimize(self.measure_transmission,self.r0)

    def set_safety_limit(self,x_safety_limit):
        self.x_safety_limit = x_safety_limit

if __name__ == "__main__":
    # try:
        o=OptimizeCoupling(x_safety_limit=1.652)
        #o.optimize_coupling()
    # except:
    #     o.Pico.__del__()
    #     o.Stage.__del__()
    #     raise