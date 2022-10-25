from scipy.optimize import minimize

class OptimizeCoupling:
    def __init__(self):
        minimize(self.measure_transmission())

    def measure_transmission(self,x_fiber,y_fiber,z_fiber):
        if x>self.range:
            raise "causion! the fiber is too close to the chip"
        stage.move_to(x_fiber,y_fiber,z_fiber)
        self.trace = scope.get_trace()
        transmission = sum(self.trace)
        return transmission

    def define_range(self,range=3):
        self.range = range
