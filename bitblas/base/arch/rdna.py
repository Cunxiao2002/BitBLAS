# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from bitblas import tvm
from tvm.target import Target
from .arch_base import TileDevice
from typing import List, Dict, Union

def check_gfx_version(arch: str)->int:
    gfx_version = arch.replace("gfx", "")
    return int(gfx_version) if gfx_version.isdigit() else -1

class RDNA(TileDevice):
    def __init__(self, target: Union[Target, str]):
        if isinstance(target, str):
            target = tvm.target.Target(target) 
        self.target = target
        self.gfx_version = check_gfx_version(self.target.arch)
        #?question: device = tvm.runtime.rocm(0)
        device = tvm.runtime.hip(0) 
        if not device.exist:
            raise RuntimeError("Cannot find HIP device 0.")
        self.device: tvm.runtime.Device = device
        self.platform: str = "RDNA"
        self.smem_cap = device.max_shared_memory_per_block
        self.compute_max_core = device.multi_processor_count
        self.warp_size = device.warp_size
        self.compute_capability = device.compute_version.replace(".", "")
        #self.compute_capability: str = "gfx90a"
        self.reg_cap: int = 32768
        self.max_smem_usage: int = 2 * self.smem_cap
        #self.max_smem_usage: int = 0
        self.sm_partition: int = 4
        self.l2_cache_size_bytes: int = target.l2_cache_size_bytes
        # the number of transaction size in bytes
        self.transaction_size: List[int] = [32, 128]  # in bytes
    
        self.bandwidth: List[int] = [1300, 14000]