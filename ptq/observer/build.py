# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
from .ema import EmaObserver
from .minmax import MinmaxObserver
from .omse import OmseObserver
from .percentile import PercentileObserver
from .ptf import *

str2observer = {
    'minmax': MinmaxObserver,
    'ema': EmaObserver,
    'omse': OmseObserver,
    'percentile': PercentileObserver,
    'ptf_0to1': PtfObserver_2_0to1,
    'ptf_0to2': PtfObserver_2_0to2,
    'ptf_0to3': PtfObserver_2_0to3,
    'ptf_0to4': PtfObserver_2_0to4,
    'ptf_0to5': PtfObserver_2_0to5,
    'ptf_0to6': PtfObserver_2_0to6,
    'ptf_0to7': PtfObserver_2_0to7,
    'ptf_0to8': PtfObserver_2_0to8,
    'ptf_0246': PtfObserver_2_0246,
    'ptf_01236': PtfObserver_2_01236,
    'ptf_0126': PtfObserver_2_0126,
    'ptf_0136': PtfObserver_2_0136,
    'ptf_0236': PtfObserver_2_0236,
    'ptf_01237': PtfObserver_2_01237,
    'ptf_0to9': PtfObserver_2_0to9,
    'ptf_0to10': PtfObserver_2_0to10,
    'ptf_0to11': PtfObserver_2_0to11,
    'ptf_0to12': PtfObserver_2_0to12,
    'ptf_0to13': PtfObserver_2_0to13,
    'ptf_0to14': PtfObserver_2_0to14,
    'ptf_0to15': PtfObserver_2_0to15,
    'ptf_0to16': PtfObserver_2_0to16,
    'ptf_3_0to10': PtfObserver_3_0to10,
    'ptf_0to17': PtfObserver_2_0to17,
    'ptf_0to18': PtfObserver_2_0to18,
    'ptf_0to19': PtfObserver_2_0to19,
    'ptf_0to20': PtfObserver_2_0to20,
}


def build_observer(observer_str, module_type, bit_type, calibration_mode):
    observer = str2observer[observer_str]
    return observer(module_type, bit_type, calibration_mode)
