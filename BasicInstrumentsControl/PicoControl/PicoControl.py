import ctypes
import numpy as np
from BasicInstrumentsControl.PicoControl.picosdk_python_wrappers.picosdk.ps4000a import ps4000a as ps
import matplotlib.pyplot as plt
from BasicInstrumentsControl.PicoControl.picosdk_python_wrappers.picosdk.functions import adc2mV, assert_pico_ok
import time
# PARAMETERS
VOLT_TO_NM = 2 # calibration of volts from sigGen to nm at the laser
ENABLED =  1

class PicoControl():
    def __init__(self):
        self.connect()

    def __del__(self):
        # Stop the scope
        # handle = chandle
        self.status["stop"] = ps.ps4000aStop(self.chandle)
        assert_pico_ok(self.status["stop"])

        # Disconnect the scope
        # handle = chandle
        self.status["close"] = ps.ps4000aCloseUnit(self.chandle)
        assert_pico_ok(self.status["close"])

        # Display status returns
        print(self.status)

    def connect(self):
        '''
        connect to pico
        :return:
        '''
        # Create chandle and status ready for use
        self.chandle = ctypes.c_int16()
        self.status = {}

        # Open PicoScope 2000 Series device
        # Returns handle to chandle for use in future API functions
        self.status["openunit"] = ps.ps4000aOpenUnit(ctypes.byref(self.chandle), None)
        try:
            assert_pico_ok(self.status["openunit"])
        except:
            powerStatus = self.status["openunit"]

            if powerStatus == 286:
                self.status["changePowerSource"] = ps.ps4000aChangePowerSource(self.chandle, powerStatus)
            else:
                raise
            assert_pico_ok(self.status["changePowerSource"])


class PicoScopeControl():
    def __init__(self,pico):
        #parameters
        self.pico = pico
        self.set_channel(channel="CH_A",channel_range = 7, analogue_offset = 0.0)
        self.set_channel(channel="CH_B", channel_range=7, analogue_offset=0.0)
        self.set_memory(sizeOfOneBuffer = 500,numBuffersToCapture = 10,Channel = "CH_A")
        self.set_memory(sizeOfOneBuffer=500, numBuffersToCapture=10, Channel="CH_B")

    def plot_trace(self):
        # Create time data
        time = 1e-9*np.linspace(0, (self.totalSamples - 1) * self.actualSampleIntervalNs, self.totalSamples)

        # Plot data from channel A and B
        plt.plot(time, self.adc2mVChAMax[:],label='Channel A')
        plt.plot(time, self.adc2mVChBMax[:], label='Channel B')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (mV)')
        plt.legend()
        plt.show()

    def get_trace(self):
        self.bufferCompleteA = np.zeros(shape=self.totalSamples, dtype=np.int16)
        self.bufferCompleteB = np.zeros(shape=self.totalSamples, dtype=np.int16)
        self.nextSample = 0
        autoStopOuter = False
        wasCalledBack = False

        # Begin streaming mode:
        sampleInterval = ctypes.c_int32(250)
        sampleUnits = ps.PS4000A_TIME_UNITS['PS4000A_US']
        # We are not triggering:
        maxPreTriggerSamples = 0
        autoStopOn = 1
        # No downsampling:
        downsampleRatio = 1
        self.pico.status["runStreaming"] = ps.ps4000aRunStreaming(self.pico.chandle,
                                                        ctypes.byref(sampleInterval),
                                                        sampleUnits,
                                                        maxPreTriggerSamples,
                                                        self.totalSamples,
                                                        autoStopOn,
                                                        downsampleRatio,
                                                        ps.PS4000A_RATIO_MODE['PS4000A_RATIO_MODE_NONE'],
                                                        self.sizeOfOneBuffer)
        assert_pico_ok(self.pico.status["runStreaming"])

        actualSampleInterval = sampleInterval.value
        self.actualSampleIntervalNs = actualSampleInterval * 1000

        print("Capturing at sample interval %s ns" % self.actualSampleIntervalNs )

        # Convert the python function into a C function pointer.
        cFuncPtr = ps.StreamingReadyType(self.streaming_callback)

        # Fetch data from the driver in a loop, copying it out of the registered buffers and into our complete one.
        while self.nextSample < self.totalSamples and not autoStopOuter:
            wasCalledBack = False
            self.pico.status["getStreamingLastestValues"] = ps.ps4000aGetStreamingLatestValues(self.pico.chandle, cFuncPtr, None)
            if not wasCalledBack:
                # If we weren't called back by the driver, this means no data is ready. Sleep for a short while before trying
                # again.
                time.sleep(0.01)

        print("Done grabbing values.")

        # Find maximum ADC count value
        # handle = chandle
        # pointer to value = ctypes.byref(maxADC)
        maxADC = ctypes.c_int16()
        self.pico.status["maximumValue"] = ps.ps4000aMaximumValue(self.pico.chandle, ctypes.byref(maxADC))
        assert_pico_ok(self.pico.status["maximumValue"])

        # Convert ADC counts data to mV
        self.adc2mVChAMax = adc2mV(self.bufferCompleteA, self.channel_range, maxADC)
        self.adc2mVChBMax = adc2mV(self.bufferCompleteB, self.channel_range, maxADC)
        return [self.adc2mVChAMax, self.adc2mVChBMax]

    def set_memory(self,sizeOfOneBuffer = 500,numBuffersToCapture = 10,Channel = "CH_A"):
        self.sizeOfOneBuffer = sizeOfOneBuffer
        self.totalSamples = self.sizeOfOneBuffer * numBuffersToCapture

        # Create buffers ready for assigning pointers for data collection
        self.bufferAMax = np.zeros(shape=sizeOfOneBuffer, dtype=np.int16)
        self.bufferBMax = np.zeros(shape=sizeOfOneBuffer, dtype=np.int16)

        memory_segment = 0
        if Channel == "CH_A":
            self.pico.status["setDataBuffersA"] = ps.ps4000aSetDataBuffers(self.pico.chandle,
                                                             ps.PS4000A_CHANNEL['PS4000A_CHANNEL_A'],
                                                             self.bufferAMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                                                             None,
                                                             sizeOfOneBuffer,
                                                             memory_segment,
                                                             ps.PS4000A_RATIO_MODE['PS4000A_RATIO_MODE_NONE'])
            assert_pico_ok(self.pico.status["setDataBuffersA"])
        else:
            self.pico.status["setDataBuffersB"] = ps.ps4000aSetDataBuffers(self.pico.chandle,
                                                                      ps.PS4000A_CHANNEL['PS4000A_CHANNEL_B'],
                                                                      self.bufferBMax.ctypes.data_as(
                                                                          ctypes.POINTER(ctypes.c_int16)),
                                                                      None,
                                                                      sizeOfOneBuffer,
                                                                      memory_segment,
                                                                      ps.PS4000A_RATIO_MODE['PS4000A_RATIO_MODE_NONE'])
            assert_pico_ok(self.pico.status["setDataBuffersB"])

    def set_channel(self, channel="CH_A",channel_range = 7, analogue_offset = 0.0):
        self.channel_range = channel_range
        if channel == "CH_A":
            self.pico.status["setChA"] = ps.ps4000aSetChannel(self.pico.chandle,
                                                    ps.PS4000A_CHANNEL['PS4000A_CHANNEL_A'],
                                                    ENABLED,
                                                    ps.PS4000A_COUPLING['PS4000A_DC'],
                                                    channel_range,
                                                    analogue_offset)
            assert_pico_ok(self.pico.status["setChA"])
        else:
            self.pico.status["setChB"] = ps.ps4000aSetChannel(self.pico.chandle,
                                                         ps.PS4000A_CHANNEL['PS4000A_CHANNEL_B'],
                                                         ENABLED,
                                                         ps.PS4000A_COUPLING['PS4000A_DC'],
                                                         channel_range,
                                                         analogue_offset)
            assert_pico_ok(self.pico.status["setChB"])

    def streaming_callback(self,handle, noOfSamples, startIndex, overflow, triggerAt, triggered, autoStop, param):
        # We need a big buffer, not registered with the driver, to keep our complete capture in.
        global autoStopOuter, wasCalledBack

        wasCalledBack = True
        destEnd = self.nextSample + noOfSamples
        sourceEnd = startIndex + noOfSamples
        self.bufferCompleteA[self.nextSample:destEnd] = self.bufferAMax[startIndex:sourceEnd]
        self.bufferCompleteB[self.nextSample:destEnd] = self.bufferBMax[startIndex:sourceEnd]
        self.nextSample += noOfSamples
        if autoStop:
            autoStopOuter = True



class PicoSigGenControl():
    def __init__(self,pico, pk_to_pk_voltage = 0.5, offset_voltage = 0, frequency = 10,wave_type = 'RAMP_UP'):
        '''

        :param pk_to_pk_voltage: voltage peak to peak of the output of the signal generator [V]
        :param offset_voltage: offset of the voltage range center from 0V
        :param frequency: repetition frequency of the signal generator [Hz]
        '''
        self.pico = pico

        #unit conversion
        self.pk_to_pk_voltage = int(pk_to_pk_voltage*1e6) #[uV]
        self.WaveType = ps.PS4000A_WAVE_TYPE['PS4000A_'+wave_type]
        self.SweepType = ps.PS4000A_SWEEP_TYPE['PS4000A_UP']
        self.TriggerType = ps.PS4000A_SIGGEN_TRIG_TYPE['PS4000A_SIGGEN_RISING']
        self.TriggerSource = ps.PS4000A_SIGGEN_TRIG_SOURCE['PS4000A_SIGGEN_NONE']
        self.extInThreshold = ctypes.c_int16(0)  # extInThreshold - Not used

        self.pico.status["SetSigGenBuiltIn"] = ps.ps4000aSetSigGenBuiltIn(self.pico.chandle, offset_voltage, self.pk_to_pk_voltage, self.WaveType, frequency, frequency, 1, 1,
                                                                self.SweepType, 0, 0, 0, self.TriggerType, self.TriggerSource,
                                                                self.extInThreshold)
        assert_pico_ok(self.pico.status["SetSigGenBuiltIn"])

    def calculate_scan_width(self):
        self.scan_width = (self.pk_to_pk_voltage*1e-6) * VOLT_TO_NM
        return self.scan_width

if __name__=='__main__':
    Pico = PicoControl()
    SigGen = PicoSigGenControl(Pico)
    Scope = PicoScopeControl(Pico)
    Scope.get_trace()
    Scope.plot_trace()
    Pico.__del__()
    # Instance = 'SCOPE'
    # if Instance== 'SCOPE':
    #     o=PicoScopeControl()
    #     o.get_trace()
    #     o.plot_trace()
    # else:
    #     o = PicoSigGenControl()