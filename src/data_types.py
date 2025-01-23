from typing import TypedDict, List

import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 300)


# endregion

# ---------------------------------------------------------------------------------------------------
# region TYPES

class LogData(TypedDict):
    type: str
    ts: int
    level: str
    msg: str
    seq: int


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


class FrzTempData(TypedDict):
    type: str
    ts: int
    left: int
    right: int
    amb: int
    hs: int
    seq: int


class BedTempChannel(TypedDict):
    side: int
    out: int
    cen: int
    in_: int  # Renamed `in` to `in_` for Python compliance


class BedTempData(TypedDict):
    type: str
    ts: int
    amb: int
    mcu: int
    hu: int
    left: BedTempChannel
    right: BedTempChannel
    seq: int


class RawRow(TypedDict):
    seq: int
    data: bytes


class Data(TypedDict):
    bed_temps: List[BedTempData]
    cap_senses: List[CapSenseData]
    freeze_temps: List[FrzTempData]
    logs: List[LogData]
    piezo_dual: List[PiezoDualData]

    errors: List[dict]





