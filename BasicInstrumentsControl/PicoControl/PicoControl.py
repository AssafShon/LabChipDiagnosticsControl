import ctypes
import numpy as np
from BasicInstrumentsControl.PicoControl.picosdk_python_wrappers.picosdk.ps4000a import ps4000a as ps
import matplotlib.pyplot as plt
from BasicInstrumentsControl.PicoControl.picosdk_python_wrappers.picosdk.functions import adc2mV, assert_pico_ok, mV2adc
import time
from math import *

# PARAMETERS
VOLT_TO_NM = 0.16/6 # calibration of volts from sigGen to nm at the laser - 1.	6 Volts pk to pk yields 0.16 nm jump(measured on the TLB screen, better result can be taken by wavelength meter).
AMP_GAIN = 6/0.8 #amplifier gain (volts to volts)
NUM_OF_SAMPLES_FOR_SINGLE_SCAN = 100 # calibrated to single scan
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
        self.set_channel(channel="CH_A",channel_range = 10, analogue_offset = 0.0)
        self.set_channel(channel="CH_B", channel_range=10, analogue_offset=0.0)
        self.set_memory(sizeOfOneBuffer = 50,numBuffersToCapture = 10,Channel = "CH_A")
        self.set_memory(sizeOfOneBuffer=50, numBuffersToCapture=10, Channel="CH_B")

    def plot_trace(self):
        # Create time data
        # Plot data from channel A and B
        plt.plot(self.time, self.adc2mVChAMax[:],label='Channel A')
        plt.plot(self.time, self.adc2mVChBMax[:], label='Channel B')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (mV)')
        plt.legend()
        plt.show()

    def set_trigger(self,trigger_threshold=10024,trigger_delay=100):
        trigger_channel = ps.PS4000A_CHANNEL['PS4000A_CHANNEL_B']
        trigger_direction = ps.PS4000A_THRESHOLD_DIRECTION["PS4000A_ABOVE"]
        self.pico.status["getTrigger"] = ps.ps4000aSetSimpleTrigger(
                                                self.pico.chandle,
                                                ENABLED,
                                                trigger_channel,
                                                trigger_threshold,
                                                trigger_direction,
                                                trigger_delay,
                                                1000000)
        assert_pico_ok(self.pico.status["getTrigger"])

    def get_trace(self):
        # Set up single trigger
        # handle = chandle
        # enabled = 1
        # source = PS4000a_CHANNEL_A = 0
        # threshold = 1024 ADC counts
        # direction = PS4000a_RISING = 2
        # delay = 0 s
        # auto Trigger = 1000 ms
        self.pico.status["trigger"] = ps.ps4000aSetSimpleTrigger(self.pico.chandle, 1, 0, 0, 2, 10000, 100)
        assert_pico_ok(self.pico.status["trigger"])

        # Set number of pre and post trigger samples to be collected
        preTriggerSamples = 220000
        postTriggerSamples = 220000
        maxSamples = preTriggerSamples + postTriggerSamples

        # Get timebase information
        # WARNING: When using this example it may not be possible to access all Timebases as all channels are enabled by default when opening the scope.
        # To access these Timebases, set any unused analogue channels to off.
        # handle = chandle
        # timebase = 8 = timebase
        # noSamples = maxSamples
        # pointer to timeIntervalNanoseconds = ctypes.byref(timeIntervalns)
        # pointer to maxSamples = ctypes.byref(returnedMaxSamples)
        # segment index = 0
        timebase = 8
        timeIntervalns = ctypes.c_float()
        returnedMaxSamples = ctypes.c_int32()
        oversample = ctypes.c_int16(1)
        self.pico.status["getTimebase2"] = ps.ps4000aGetTimebase2(self.pico.chandle, timebase, maxSamples, ctypes.byref(timeIntervalns),
                                                        ctypes.byref(returnedMaxSamples), 0)
        assert_pico_ok(self.pico.status["getTimebase2"])

        # Run block capture
        # handle = chandle
        # number of pre-trigger samples = preTriggerSamples
        # number of post-trigger samples = PostTriggerSamples
        # timebase = 3 = 80 ns = timebase (see Programmer's guide for mre information on timebases)
        # time indisposed ms = None (not needed in the example)
        # segment index = 0
        # lpReady = None (using ps4000aIsReady rather than ps4000aBlockReady)
        # pParameter = None
        self.pico.status["runBlock"] = ps.ps4000aRunBlock(self.pico.chandle, preTriggerSamples, postTriggerSamples, timebase, None, 0, None,
                                                None)
        assert_pico_ok(self.pico.status["runBlock"])

        # Check for data collection to finish using ps4000aIsReady
        ready = ctypes.c_int16(0)
        check = ctypes.c_int16(0)
        while ready.value == check.value:
            self.pico.status["isReady"] = ps.ps4000aIsReady(self.pico.chandle, ctypes.byref(ready))

        # Create buffers ready for assigning pointers for data collection
        self.bufferAMax = (ctypes.c_int16 * maxSamples)()
        self.bufferAMin = (ctypes.c_int16 * maxSamples)()  # used for downsampling which isn't in the scope of this example
        self.bufferBMax = (ctypes.c_int16 * maxSamples)()
        self.bufferBMin = (ctypes.c_int16 * maxSamples)()  # used for downsampling which isn't in the scope of this example

        # Set data buffer location for data collection from channel A
        # handle = chandle
        # source = PS4000a_CHANNEL_A = 0
        # pointer to buffer max = ctypes.byref(bufferAMax)
        # pointer to buffer min = ctypes.byref(bufferAMin)
        # buffer length = maxSamples
        # segementIndex = 0
        # mode = PS4000A_RATIO_MODE_NONE = 0
        self.pico.status["setDataBuffersA"] = ps.ps4000aSetDataBuffers(self.pico.chandle, 0, ctypes.byref(self.bufferAMax),
                                                             ctypes.byref(self.bufferAMin), maxSamples, 0, 0)
        assert_pico_ok(self.pico.status["setDataBuffersA"])

        # Set data buffer location for data collection from channel B
        # handle = chandle
        # source = PS4000a_CHANNEL_B = 1
        # pointer to buffer max = ctypes.byref(bufferBMax)
        # pointer to buffer min = ctypes.byref(bufferBMin)
        # buffer length = maxSamples
        # segementIndex = 0
        # mode = PS4000A_RATIO_MODE_NONE = 0
        self.pico.status["setDataBuffersB"] = ps.ps4000aSetDataBuffers(self.pico.chandle, 1, ctypes.byref(self.bufferBMax),
                                                             ctypes.byref(self.bufferBMin), maxSamples, 0, 0)
        assert_pico_ok(self.pico.status["setDataBuffersB"])

        # create overflow loaction
        overflow = ctypes.c_int16()
        # create converted type maxSamples
        cmaxSamples = ctypes.c_int32(maxSamples)

        # Retried data from scope to buffers assigned above
        # handle = chandle
        # start index = 0
        # pointer to number of samples = ctypes.byref(cmaxSamples)
        # downsample ratio = 0
        # downsample ratio mode = PS4000a_RATIO_MODE_NONE
        # pointer to overflow = ctypes.byref(overflow))
        self.pico.status["getValues"] = ps.ps4000aGetValues(self.pico.chandle, 0, ctypes.byref(cmaxSamples), 0, 0, 0,
                                                  ctypes.byref(overflow))
        assert_pico_ok(self.pico.status["getValues"])

        # find maximum ADC count value
        # handle = chandle
        # pointer to value = ctypes.byref(maxADC)
        maxADC = ctypes.c_int16(32767)

        # convert ADC counts data to mV
        self.adc2mVChAMax = adc2mV(self.bufferAMax,self.chARange, maxADC)
        self.adc2mVChBMax = adc2mV(self.bufferBMax, self.chBRange, maxADC)

        # Create time data
        self.time = np.linspace(0, (cmaxSamples.value - 1) * timeIntervalns.value, cmaxSamples.value)

        return [self.adc2mVChAMax, self.adc2mVChBMax]

    def set_memory(self,sizeOfOneBuffer = 500,numBuffersToCapture = 10,Channel = "CH_A"):
        self.sizeOfOneBuffer = sizeOfOneBuffer
        self.totalSamples = self.sizeOfOneBuffer * numBuffersToCapture

        # Create buffers ready for assigning pointers for data collection


        if Channel == "CH_A":
            self.bufferAMax = np.zeros(shape=sizeOfOneBuffer, dtype=np.int16)
            memory_segment = 0

            self.pico.status["setDataBuffersA"] = ps.ps4000aSetDataBuffers(self.pico.chandle,
                                                             ps.PS4000A_CHANNEL['PS4000A_CHANNEL_A'],
                                                             self.bufferAMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                                                             None,
                                                             sizeOfOneBuffer,
                                                             memory_segment,
                                                             ps.PS4000A_RATIO_MODE['PS4000A_RATIO_MODE_NONE'])
            assert_pico_ok(self.pico.status["setDataBuffersA"])
        else:
            self.bufferBMax = np.zeros(shape=sizeOfOneBuffer, dtype=np.int16)
            memory_segment = 0

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
            self.chARange = channel_range
        else:
            self.pico.status["setChB"] = ps.ps4000aSetChannel(self.pico.chandle,
                                                         ps.PS4000A_CHANNEL['PS4000A_CHANNEL_B'],
                                                         ENABLED,
                                                         ps.PS4000A_COUPLING['PS4000A_DC'],
                                                         channel_range,
                                                         analogue_offset)
            assert_pico_ok(self.pico.status["setChB"])
            self.chBRange = channel_range

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
    def __init__(self,pico, pk_to_pk_voltage = 0.8, offset_voltage = 0, frequency = 10,wave_type = 'TRIANGLE'):
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
        self.scan_width = (self.pk_to_pk_voltage*1e-6*AMP_GAIN) * VOLT_TO_NM
        return self.scan_width

if __name__=='__main__':
    Pico = PicoControl()
    SigGen = PicoSigGenControl(Pico)
    Scope = PicoScopeControl(Pico)
    # Scope.set_trigger()
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