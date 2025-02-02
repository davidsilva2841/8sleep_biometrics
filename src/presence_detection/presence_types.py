import numpy as np
from typing import TypedDict, Union, Literal, List

Side = Literal['left', 'right']


class Baseline(TypedDict):
    mean: float
    std: float

# Capacitance baseline
class CapBaseline(TypedDict):
    left_out: Baseline
    left_cen: Baseline
    left_in: Baseline
    right_out: Baseline
    right_cen: Baseline
    right_in: Baseline


class PiezoDualData(TypedDict):
    type: str
    ts: int
    freq: int
    adc: int
    gain: int
    left1: np.ndarray
    left2: np.ndarray
    right1: np.ndarray
    right2: np.ndarray
    seq: int



class CapSenseChannel(TypedDict):
    out: int
    cen: int
    in_: int  # Renamed `in` to `in_` for Python compliance
    status: str


class CapSenseData(TypedDict):
    type: str
    ts: int
    left: CapSenseChannel
    right: CapSenseChannel
    seq: int


class Data(TypedDict):
    cap_senses: List[CapSenseData]
    piezo_dual: List[PiezoDualData]


