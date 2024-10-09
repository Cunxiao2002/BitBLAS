# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from bitblas import tvm
from tvm.target import Target
from .arch_base import TileDevice
from typing import List, Dict, Union


class MI250(TileDevice):
    def __init__(self):
        self.reg_cap: int = 32768
        self.smem_cap: int = 65536
        self.compute_max_core: int = 104
        self.warp_size: int = 64
        self.sm_partition: int = 4
        self.transaction_size: List[int] = [32, 128]
        self.max_smem_usage: int = 0
        self.bandwidth: List[int] = [1300, 14000]
        self.platform: str = "ROCm-CDNA2"
        self.compute_capability: str = "gfx90a"
        self.target = tvm.target.Target("hip --mcpu=gfx90a")