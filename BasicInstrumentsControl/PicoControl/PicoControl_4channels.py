import ctypes
import numpy as np
from picosdk.ps3000a import ps3000a as ps

from picosdk.functions import assert_pico_ok

#from BasicInstrumentsControl.PicoControl.picosdk_python_wrappers.picosdk.ps3000a import ps3000a as ps
import matplotlib.pyplot as plt
from BasicInstrumentsControl.PicoControl.picosdk_python_wrappers.picosdk.functions import adc2mV, assert_pico_ok, mV2adc
import time
from math import *

# PARAMETERS
VOLT_TO_NM = 0.17/6 # calibration of volts from sigGen to nm at the laser - 1.	6 Volts pk to pk yields 0.16 nm jump(measured on the TLB screen, better result can be taken by wavelength meter).
AMP_GAIN = 6/0.8 #amplifier gain (volts to volts)
NUM_OF_SAMPLES_FOR_SINGLE_SCAN = 100 # calibrated to single scan
ENABLED = 1
CH_A = 0
CH_B = 1


class PicoControl():
    def __init__(self):
        self.connect()
        self.dictionary_voltage_range = ps.PICO_VOLTAGE_RANGE # voltage range dictionary for the scope

    def __del__(self):
        # Stop the scope
        # handle = chandle
        self.status["stop"] = ps.ps3000aStop(self.chandle)
        assert_pico_ok(self.status["stop"])

        # Disconnect the scope
        # handle = chandle
        self.status["close"] = ps.ps3000aCloseUnit(self.chandle)
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
        self.status["openunit"] = ps.ps3000aOpenUnit(ctypes.byref(self.chandle), None)
        try:
            assert_pico_ok(self.status["openunit"])
        except:
            powerStatus = self.status["openunit"]

            if powerStatus == 286:
                self.status["changePowerSource"] = ps.ps3000aChangePowerSource(self.chandle, powerStatus)
            else:
                raise
            assert_pico_ok(self.status["changePowerSource"])


class PicoScopeControl():
    def __init__(self,pico):
        #parameters
        self.pico = pico
        self.set_channel(channel="CH_A",channel_range = 8, analog_offset = 0.0)
        self.set_channel(channel="CH_B", channel_range=5, analog_offset=0.0)
        self.set_channel(channel="CH_C", channel_range=5, analog_offset=0.0)
        self.set_memory(sizeOfOneBuffer = 50,numBuffersToCapture = 10,Channel = "CH_A")
        self.set_memory(sizeOfOneBuffer=50, numBuffersToCapture=10, Channel="CH_B")
        self.set_memory(sizeOfOneBuffer=50, numBuffersToCapture=10, Channel="CH_C")

    def plot_trace(self):
        # Create time data
        # Plot data from channel A and B
        # plt.ylim(500,-500)
        # plt.plot(self.time, self.adc2mVChAMax[:],label='Channel A')
        plt.plot(self.time, self.adc2mVChAMax, label='Channel A')
        plt.plot(self.time, self.adc2mVChBMax, label='Channel B')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (mV)')
        plt.legend()
        plt.show()

    def set_trigger(self, trigger_direction = ps.PS3000A_THRESHOLD_DIRECTION['PS3000A_RISING'],thresholdUpper=261,
                    hysteresisUpper=1000, thresholdLower=-10240, thresholdLowerHysteresis=1024,channel =  ps.PS3000A_CHANNEL['PS3000A_CHANNEL_A'],
                    thresholdMode =ps.PS3000A_THRESHOLD_MODE['PS3000A_LEVEL'], nChannelProperties = 1, autoTriggerMilliseconds = 10000):
        '''
        :param thresholdUpper: the upper threshold at which the trigger must fire. This is scaled in 16-bit ADC counts at the currently selected range for that channel.
        :param hysteresisUpper:  the distance by which the signal must fall below the upper threshold (for rising edge triggers) or rise above the upper threshold (for falling edge triggers) in order to rearm the trigger for the next event. It is scaled in 16-bit counts.
        :param thresholdLower:   the lower threshold at which the trigger must fire. This is scaled in 16-bit ADC counts at the currently selected range for that channel.
        :param thresholdLowerHysteresis:  the hysteresis by which the trigger must exceed the lower threshold before it will fire. It is scaled in 16-bit counts.
        :param thresholdMode: either a level or window trigger. Use one of these constants: LEVEL\WINDOW
        :param nChannelProperties: the size of the channelProperties array. If zero, triggering is switched off
        :return:
        '''
        # self.pico.status["trigger"] = ps.ps3000aSetSimpleTrigger(self.pico.chandle, 1, 0, 0, 2, 10000, 100)

        # PS3000A_DIRECTION = [ps.PS3000A_DIRECTION(channel,trigger_direction,thresholdMode)]
        #defining channel directions
        chA_trigger_direction = trigger_direction
        chB_trigger_direction = trigger_direction
        chC_trigger_direction = trigger_direction
        chD_trigger_direction = trigger_direction
        ch_ext_trigger_direction = trigger_direction
        ch_aux_trigger_direction = trigger_direction

        state_true = ps.PS3000A_TRIGGER_STATE["PS3000A_CONDITION_FALSE"]
        state_dont_care = ps.PS3000A_TRIGGER_STATE["PS3000A_CONDITION_DONT_CARE"]
        chA_trigger_condition = state_true    # only Channel A have active trigger, this channel get the sigGen signal
        chB_trigger_condition = state_dont_care
        chC_trigger_condition = state_dont_care
        chD_trigger_condition = state_dont_care
        ch_ext_trigger_condition = state_dont_care
        ch_aux_trigger_condition = state_dont_care

        self.pico.status["trigger"] = ps.ps3000aSetTriggerChannelDirections(self.pico.chandle, chA_trigger_direction,
                    chB_trigger_direction,chC_trigger_direction,chD_trigger_direction,ch_ext_trigger_direction,
                                                                            ch_aux_trigger_direction)
        assert_pico_ok(self.pico.status["trigger"])

        PS3000A_TRIGGER_CHANNEL_PROPERTIES = ps.PS3000A_TRIGGER_CHANNEL_PROPERTIES(thresholdUpper, hysteresisUpper,
                                                                                   thresholdLower, thresholdLowerHysteresis,
                                                                                   channel, thresholdMode)
        self.pico.status["setTrigProp"] = ps.ps3000aSetTriggerChannelProperties(self.pico.chandle, ctypes.byref(
            PS3000A_TRIGGER_CHANNEL_PROPERTIES), nChannelProperties, 0, autoTriggerMilliseconds)
        assert_pico_ok(self.pico.status["setTrigProp"])

        nConditions = 1


        trigConditionA = ps.PS3000A_TRIGGER_CONDITIONS(chA_trigger_condition,chB_trigger_condition,chC_trigger_condition,
                                                    chD_trigger_condition,ch_ext_trigger_condition,ch_aux_trigger_condition,
                                                       state_true)
        assert_pico_ok(ps.ps3000aSetTriggerChannelConditions(self.pico.chandle,
                                                        ctypes.byref(trigConditionA), nConditions))





    def set_trigger_backup(self, trigger_direction=ps.PS3000A_THRESHOLD_DIRECTION['PS3000A_RISING'], thresholdUpper=1000,
                    hysteresisUpper=1000, thresholdLower=-10240, thresholdLowerHysteresis=1024,
                    channel=ps.PS3000A_CHANNEL['PS3000A_CHANNEL_A'],
                    thresholdMode=ps.PS3000A_THRESHOLD_MODE['PS3000A_LEVEL'], nChannelProperties=1,
                    autoTriggerMilliseconds=10000):
        '''
        :param thresholdUpper: the upper threshold at which the trigger must fire. This is scaled in 16-bit ADC counts at the currently selected range for that channel.
        :param hysteresisUpper:  the distance by which the signal must fall below the upper threshold (for rising edge triggers) or rise above the upper threshold (for falling edge triggers) in order to rearm the trigger for the next event. It is scaled in 16-bit counts.
        :param thresholdLower:   the lower threshold at which the trigger must fire. This is scaled in 16-bit ADC counts at the currently selected range for that channel.
        :param thresholdLowerHysteresis:  the hysteresis by which the trigger must exceed the lower threshold before it will fire. It is scaled in 16-bit counts.
        :param thresholdMode: either a level or window trigger. Use one of these constants: LEVEL\WINDOW
        :param nChannelProperties: the size of the channelProperties array. If zero, triggering is switched off
        :return:
        '''
        # self.pico.status["trigger"] = ps.ps3000aSetSimpleTrigger(self.pico.chandle, 1, 0, 0, 2, 10000, 100)

        # PS3000A_DIRECTION = [ps.PS3000A_DIRECTION(channel,trigger_direction,thresholdMode)]
        # defining channel directions
        chA_trigger_direction = trigger_direction
        chB_trigger_direction = trigger_direction
        chC_trigger_direction = trigger_direction
        chD_trigger_direction = trigger_direction
        ch_ext_trigger_direction = trigger_direction
        ch_aux_trigger_direction = trigger_direction

        state_true = ps.PS3000A_TRIGGER_STATE["PS3000A_CONDITION_FALSE"]
        state_dont_care = ps.PS3000A_TRIGGER_STATE["PS3000A_CONDITION_DONT_CARE"]
        chA_trigger_condition = state_true  # only Channel A have active trigger, this channel get the sigGen signal
        chB_trigger_condition = state_dont_care
        chC_trigger_condition = state_dont_care
        chD_trigger_condition = state_dont_care
        ch_ext_trigger_condition = state_dont_care
        ch_aux_trigger_condition = state_dont_care

        self.pico.status["trigger"] = ps.ps3000aSetTriggerChannelDirections(self.pico.chandle, chA_trigger_direction,
                                                                            chB_trigger_direction,
                                                                            chC_trigger_direction,
                                                                            chD_trigger_direction,
                                                                            ch_ext_trigger_direction,
                                                                            ch_aux_trigger_direction)
        assert_pico_ok(self.pico.status["trigger"])

        PS3000A_TRIGGER_CHANNEL_PROPERTIES = ps.PS3000A_TRIGGER_CHANNEL_PROPERTIES(thresholdUpper, hysteresisUpper,
                                                                                   thresholdLower,
                                                                                   thresholdLowerHysteresis,
                                                                                   channel, thresholdMode)
        self.pico.status["setTrigProp"] = ps.ps3000aSetTriggerChannelProperties(self.pico.chandle, ctypes.byref(
            PS3000A_TRIGGER_CHANNEL_PROPERTIES), nChannelProperties, 0, autoTriggerMilliseconds)
        assert_pico_ok(self.pico.status["setTrigProp"])

        nConditions = 1

        trigConditionA = ps.PS3000A_TRIGGER_CONDITIONS(chA_trigger_condition, chB_trigger_condition,
                                                       chC_trigger_condition,
                                                       chD_trigger_condition, ch_ext_trigger_condition,
                                                       ch_aux_trigger_condition,
                                                       state_true)
        assert_pico_ok(ps.ps3000aSetTriggerChannelConditions(self.pico.chandle,
                                                             ctypes.byref(trigConditionA), nConditions))  #

    def calibrate_range(self):
        '''
        calibrates the range of a trace to get the maximal resolution.
        :return: the range of values of the trace
        '''
        #take trace for calibration (in V)
        self.set_channel(channel="CH_B", channel_range=5)
        self.calibrate_trace = np.array(self.get_trace()[CH_B])
        # find range which is one after the closest one to the data (to make sure the limit is not
        # too close)
        self.trace_range = np.ptp(self.calibrate_trace)

        # finds the channel range which is larger by one from the closest one to the signal range
        range_diff = np.array(list(self.pico.dictionary_voltage_range.values()))*1000 - self.trace_range
        self.channel_range = int(np.where(range_diff == [min(range_diff[range_diff > 0])])[0])    #+1

        # finds the min/max analog offset for specific offset
        self.maxAnalogOffset = ctypes.c_float()
        self.minAnalogOffset = ctypes.c_float()
        ps.ps3000aGetAnalogueOffset(self.pico.chandle, self.channel_range, ps.PS3000A_COUPLING['PS3000A_DC'],
                             ctypes.byref(self.maxAnalogOffset), ctypes.byref(self.minAnalogOffset))
        self.maxAnalogOffset.value = self.maxAnalogOffset.value * 1000
        self.minAnalogOffset.value = self.minAnalogOffset.value * 1000

        # find mean value
        self.calibrate_trace_avg_voltage = float(np.mean(self.calibrate_trace))

        # if the signal dc is less than mA don't add offset
        if np.abs(self.calibrate_trace_avg_voltage) < 1:
            self.analog_offset = 0
        # if the offset is bigger than the max value possible
        elif self.calibrate_trace_avg_voltage > self.maxAnalogOffset.value:
            self.analog_offset = self.maxAnalogOffset.value
        else:
            self.analog_offset = np.max([self.minAnalogOffset.value, self.calibrate_trace_avg_voltage])

        self.set_channel(channel="CH_B",channel_range=self.channel_range, analog_offset=-self.analog_offset/1000)


    def get_trace(self):
        '''
        calls for trace from the pico, triggered by upper and lower threshold to handle noise in trigger channel.
        :return:
        '''
        # Set up single trigger
        # handle = chandle
        # enabled = 1
        # source = PS3000a_CHANNEL_A = 0
        # threshold = 1024 ADC counts
        # direction = PS3000a_RISING = 2
        # delay = 0 s
        # auto Trigger = 1000 ms
        maxADC = ctypes.c_int16(32767)
        self.set_trigger()

        # Set number of pre and post trigger samples to be collected. Calibrated for 10 Hz scan rate of the sigGen.
        preTriggerSamples = 40800 # 40500
        postTriggerSamples = 40800 # 40500
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
        timebase = 80
        timeIntervalns = ctypes.c_float()
        returnedMaxSamples = ctypes.c_int32()
        self.pico.status["getTimebase2"] = ps.ps3000aGetTimebase2(self.pico.chandle, timebase, maxSamples, ctypes.byref(timeIntervalns),0,
                                                        ctypes.byref(returnedMaxSamples), 0)
        assert_pico_ok(self.pico.status["getTimebase2"])

        # Run block capture
        # handle = chandle
        # number of pre-trigger samples = preTriggerSamples
        # number of post-trigger samples = PostTriggerSamples
        # timebase = 3 = 80 ns = timebase (see Programmer's guide for mre information on timebases)
        # time indisposed ms = None (not needed in the example)
        # segment index = 0
        # lpReady = None (using ps3000aIsReady rather than ps3000aBlockReady)
        # pParameter = None
        self.pico.status["runBlock"] = ps.ps3000aRunBlock(self.pico.chandle, preTriggerSamples, postTriggerSamples, timebase, 0, None, 0, None,
                                                None)
        assert_pico_ok(self.pico.status["runBlock"])

        # Check for data collection to finish using ps3000aIsReady
        ready = ctypes.c_int16(0)
        check = ctypes.c_int16(0)
        while ready.value == check.value:
            self.pico.status["isReady"] = ps.ps3000aIsReady(self.pico.chandle, ctypes.byref(ready))

        # Create buffers ready for assigning pointers for data collection
        self.bufferAMax = (ctypes.c_int16 * maxSamples)()
        self.bufferAMin = (ctypes.c_int16 * maxSamples)()  # used for downsampling which isn't in the scope of this example
        self.bufferBMax = (ctypes.c_int16 * maxSamples)()
        self.bufferBMin = (ctypes.c_int16 * maxSamples)()  # used for downsampling which isn't in the scope of this example
        self.bufferCMax = (ctypes.c_int16 * maxSamples)()
        self.bufferCMin = (ctypes.c_int16 * maxSamples)()  # used for downsampling which isn't in the scope of this example

        # Set data buffer location for data collection from channel A
        # handle = chandle
        # source = PS4000a_CHANNEL_A = 0
        # pointer to buffer max = ctypes.byref(bufferAMax)
        # pointer to buffer min = ctypes.byref(bufferAMin)
        # buffer length = maxSamples
        # segementIndex = 0
        # mode = PS4000A_RATIO_MODE_NONE = 0
        self.pico.status["setDataBuffersA"] = ps.ps3000aSetDataBuffers(self.pico.chandle, 0, ctypes.byref(self.bufferAMax),
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
        self.pico.status["setDataBuffersB"] = ps.ps3000aSetDataBuffers(self.pico.chandle, 1, ctypes.byref(self.bufferBMax),
                                                             ctypes.byref(self.bufferBMin), maxSamples, 0, 0)
        assert_pico_ok(self.pico.status["setDataBuffersB"])

        # Set data buffer location for data collection from channel A
        # handle = chandle
        # source = PS4000a_CHANNEL_A = 0
        # pointer to buffer max = ctypes.byref(bufferAMax)
        # pointer to buffer min = ctypes.byref(bufferAMin)
        # buffer length = maxSamples
        # segementIndex = 0
        # mode = PS4000A_RATIO_MODE_NONE = 0
        self.pico.status["setDataBuffersC"] = ps.ps3000aSetDataBuffers(self.pico.chandle, 2, ctypes.byref(self.bufferCMax),
                                                             ctypes.byref(self.bufferCMin), maxSamples, 0, 0)
        assert_pico_ok(self.pico.status["setDataBuffersC"])

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
        self.pico.status["getValues"] = ps.ps3000aGetValues(self.pico.chandle, 0, ctypes.byref(cmaxSamples), 0, 0, 0,
                                                  ctypes.byref(overflow))
        assert_pico_ok(self.pico.status["getValues"])

        # find maximum ADC count value
        # handle = chandle
        # pointer to value = ctypes.byref(maxADC)

        # convert ADC counts data to mV
        self.adc2mVChAMax = adc2mV(self.bufferAMax,self.chARange, maxADC)
        self.adc2mVChBMax = adc2mV(self.bufferBMax, self.chBRange, maxADC)
        self.adc2mVChCMax = adc2mV(self.bufferCMax, self.chCRange, maxADC)
        # Create time data
        self.time = np.linspace(0, (cmaxSamples.value - 1) * timeIntervalns.value, cmaxSamples.value)

        return [self.adc2mVChAMax, self.adc2mVChBMax,self.adc2mVChCMax]

    def set_memory(self,sizeOfOneBuffer = 500,numBuffersToCapture = 10,Channel = "CH_A"):
        self.sizeOfOneBuffer = sizeOfOneBuffer
        self.totalSamples = self.sizeOfOneBuffer * numBuffersToCapture

        # Create buffers ready for assigning pointers for data collection


        if Channel == "CH_A":
            self.bufferAMax = np.zeros(shape=sizeOfOneBuffer, dtype=np.int16)
            memory_segment = 0

            self.pico.status["setDataBuffersA"] = ps.ps3000aSetDataBuffers(self.pico.chandle,
                                                             ps.PS3000A_CHANNEL['PS3000A_CHANNEL_A'],
                                                             self.bufferAMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                                                             None,
                                                             sizeOfOneBuffer,
                                                             memory_segment,
                                                             ps.PS3000A_RATIO_MODE['PS3000A_RATIO_MODE_NONE'])
            assert_pico_ok(self.pico.status["setDataBuffersA"])

        if Channel == "CH_B":
            self.bufferBMax = np.zeros(shape=sizeOfOneBuffer, dtype=np.int16)
            memory_segment = 0

            self.pico.status["setDataBuffersB"] = ps.ps3000aSetDataBuffers(self.pico.chandle,
                                                                      ps.PS3000A_CHANNEL['PS3000A_CHANNEL_B'],
                                                                      self.bufferBMax.ctypes.data_as(
                                                                          ctypes.POINTER(ctypes.c_int16)),
                                                                      None,
                                                                      sizeOfOneBuffer,
                                                                      memory_segment,
                                                                      ps.PS3000A_RATIO_MODE['PS3000A_RATIO_MODE_NONE'])
            assert_pico_ok(self.pico.status["setDataBuffersB"])

        else:
            self.bufferBMax = np.zeros(shape=sizeOfOneBuffer, dtype=np.int16)
            memory_segment = 0

            self.pico.status["setDataBuffersC"] = ps.ps3000aSetDataBuffers(self.pico.chandle,
                                                                           ps.PS3000A_CHANNEL['PS3000A_CHANNEL_C'],
                                                                           self.bufferBMax.ctypes.data_as(
                                                                               ctypes.POINTER(ctypes.c_int16)),
                                                                           None,
                                                                           sizeOfOneBuffer,
                                                                           memory_segment,
                                                                           ps.PS3000A_RATIO_MODE['PS3000A_RATIO_MODE_NONE'])
            assert_pico_ok(self.pico.status["setDataBuffersC"])

    def set_channel(self, channel="CH_A",channel_range=7, analog_offset=0.0):
        '''

        :param channel: channel a ("CH_A") or b ("CH_B") or c ("CH_C")
        :param channel_range: voltage range - table of range per number in API (2 - 50mv, 8 - 5V). Affects the digitization resolution.
        :param analogue_offset: an offset, in volts, to be added to the input signal before it reaches the input amplifier and digitizer.
                                up to 250mv - See the device data sheet for the allowable range.
        :return:
        '''
        if channel == "CH_A":
            self.pico.status["setChA"] = ps.ps3000aSetChannel(self.pico.chandle,
                                                    ps.PS3000A_CHANNEL['PS3000A_CHANNEL_A'],
                                                    ENABLED,
                                                    ps.PS3000A_COUPLING['PS3000A_DC'],
                                                    channel_range,
                                                    analog_offset)
            assert_pico_ok(self.pico.status["setChA"])
            self.chARange = channel_range
        if channel == "CH_B":
            self.pico.status["setChB"] = ps.ps3000aSetChannel(self.pico.chandle,
                                                         ps.PS3000A_CHANNEL['PS3000A_CHANNEL_B'],
                                                         ENABLED,
                                                         ps.PS3000A_COUPLING['PS3000A_DC'],
                                                         channel_range,
                                                         analog_offset)
            assert_pico_ok(self.pico.status["setChB"])
            self.chBRange = channel_range
        else:
            self.pico.status["setChC"] = ps.ps3000aSetChannel(self.pico.chandle,
                                                         ps.PS3000A_CHANNEL['PS3000A_CHANNEL_C'],
                                                         ENABLED,
                                                         ps.PS3000A_COUPLING['PS3000A_DC'],
                                                         channel_range,
                                                         analog_offset)
            assert_pico_ok(self.pico.status["setChC"])
            self.chCRange = channel_range

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
    def __init__(self,pico, pk_to_pk_voltage = 0.8, offset_voltage = 0, frequency = 9.9, wave_type = 'TRAINGLE'):  # frequency = 10
        '''

        :param pk_to_pk_voltage: voltage peak to peak of the output of the signal generator [V].
                                 With the current amplifier should be 0.8V to generate 6V pk to pk which is the laser dynamic range.
        :param offset_voltage: offset of the voltage range center from 0V
        :param frequency: repetition frequency of the signal generator [Hz]. Change can harm TransmissionSpectrum run - Any change in the frequency affect the number of repetitions of the traces from each scan.
        '''
        self.pico = pico

        #unit conversion
        self.pk_to_pk_voltage = int(pk_to_pk_voltage*1e6) #[uV]
        # self.WaveType = ps.PS4000A_WAVE_TYPE['PS3000A_'+wave_type]
        self.WaveType = ps.PS3000A_WAVE_TYPE['PS3000A_TRIANGLE']
        # self.WaveType = ps.ctypes.c_int16(0)
        self.SweepType = ps.PS3000A_SWEEP_TYPE['PS3000A_UP']
        self.TriggerType = ps.PS3000A_SIGGEN_TRIG_TYPE['PS3000A_SIGGEN_RISING']
        self.TriggerSource = ps.PS3000A_SIGGEN_TRIG_SOURCE['PS3000A_SIGGEN_NONE']
        self.extInThreshold = ctypes.c_int16(0)  # extInThreshold - Not used

        # self.pico.status["SetSigGenBuiltIn"] = ps.ps3000aSetSigGenBuiltIn(self.pico.chandle, 0, 2000000, self.WaveType, 10000, 10000, 0, 1,
        #                                                         self.SweepType, 0, 0, 0, self.TriggerType, self.TriggerSource, 1)
        # assert_pico_ok(self.pico.status["SetSigGenBuiltIn"])
        self.pico.status["SetSigGenBuiltIn"] = ps.ps3000aSetSigGenBuiltIn(self.pico.chandle, offset_voltage, self.pk_to_pk_voltage, self.WaveType, frequency, frequency, 1, 1,
                                                               self.SweepType, 0, 0, 0, self.TriggerType, self.TriggerSource,
                                                               self.extInThreshold)
        assert_pico_ok(self.pico.status["SetSigGenBuiltIn"])

    def calculate_scan_width(self):
        '''
        calculate the scan width in nm per pk to pk voltage.
        :return:
        '''
        self.scan_width = (self.pk_to_pk_voltage*1e-6*AMP_GAIN) * VOLT_TO_NM
        return self.scan_width

if __name__=='__main__':
    Pico = PicoControl()
    SigGen = PicoSigGenControl(Pico)
    Scope = PicoScopeControl(Pico)
    # i=0
    # for i in range(2):
    Scope.get_trace()
    Scope.plot_trace()
        # i=i+1

    # Pico.__del__()
    # Instance = 'SCOPE'
    # if Instance== 'SCOPE':
    #     o=PicoScopeControl()
    #     o.get_trace()
    #     o.plot_trace()
    # else:
    #     o = PicoSigGenControl()