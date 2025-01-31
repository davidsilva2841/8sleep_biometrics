from typing import TypedDict, List

import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 300)


# endregion

# ---------------------------------------------------------------------------------------------------
# See all the different logs in ./decoded_raw_data_samples/logs.json

class LogData(TypedDict):
    type: str
    ts: int
    level: str
    msg: str
    seq: int



# ---------------------------------------------------------------------------------------------------
# Piezo sensor data (used for heart rates, hrv - ???)

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

# {
#     "adc": 1,
#     "freq": 500,
#     "gain": 400,
#     "left1": [
#         -160889, -163532, -161494, -162596, -163266, -163120, -163281,   ...... (500 values)
#     ],
#     "left2": [
#         -4788, -4841, -7013, -6902, -9195, -9662, -11273, -9883, -11415, ...... (500 values)
#     ],
#     "right1": [
#         544338, 543290, 540837, 541583, 541184, 539035, 538201, 537129,  ...... (500 values)
#     ],
#     "right2": [
#         722955, 723792, 724770, 727022, 727501, 728404, 728542, 728296,  ...... (500 values)
#     ],
#     "seq": 1610681,
#     "ts": "2025-01-10 11:00:22",
#     "type": "piezo-dual"
# }

# ---------------------------------------------------------------------------------------------------
# Capacitance sensor - Used for presence detection (I think)

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

# {
#   "type": "capSense",
#   "ts": "2025-01-10 11:00:22",
#   "left": {
#     "out": 387,
#     "cen": 381,
#     "in": 505,
#     "status": "good"
#   },
#   "right": {
#     "out": 1076,
#     "cen": 1075,
#     "in": 1074,
#     "status": "good"
#   },
#   "seq": 1610679
# }

# ---------------------------------------------------------------------------------------------------
# Freeze temps - Temperature measurements

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

# {
#     "amb": 2168,
#     "hs": 3168,
#     "left": 1975,
#     "right": 1981,
#     "seq": 1610686,
#     "ts": "2025-01-10 11:00:28",
#     "type": "frzTemp"
# }

# ---------------------------------------------------------------------------------------------------



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





