# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from abc import ABC, abstractmethod
from bitblas import tvm
from tvm import IRModule
from tvm.target import Target
from tvm.tir import PrimFunc
from tvm.contrib.dlpack import to_pytorch_func
import bitblas
import ctypes
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from bitblas.base import fast_tune, fast_tune_with_dynamic_range
from copy import deepcopy
from bitblas.base.arch import get_arch
from bitblas.base.roller.hint import Hint
from bitblas.builder.wrapper import TIRWrapper
from bitblas.builder.lib_generator import LibraryGenerator
from dataclasses import dataclass
from enum import IntEnum
import logging

logger = logging.getLogger(__name__)

APPLY_SCHEDULE_FAILED_MESSAGE = ("Failed to apply default schedule for operator {} "
                                 "With target {} and hint {}. \n"
                                 "The error message: {} "
                                 "Please perform hardware-aware tuning manually.")

BUILD_RUNTIME_LIBRARY_FAILED_MESSAGE = ("Failed to build runtime library for operator {} "
                                        "With target {} and hint {}. \n"
                                        "The error message: {} "
                                        "Please perform hardware-aware tuning manually.")


class TransformKind(IntEnum):
    NonTransform = 0
    InterWarpTransform = 1
    IntraWarpTransform = 2
    LDMatrixTransform = 3


@dataclass(frozen=True)
class OperatorConfig:
    """Base class for operator configurations. Used for typing."""
    pass


class BaseKernelNameGenerator(ABC):
    """Optional class for generating kernel names based on the config and hint"""

    def __init__(self, config: OperatorConfig):
        assert self.is_valid_config(config), (f"Invalid config for {self.__class__.__name__}: "
                                              f"{config}")
        self.config = config

    @abstractmethod
    def is_valid_config(self, config: OperatorConfig):
        pass

    @abstractmethod
    def generate(self, hint: Hint = None) -> str:
        '''Generate the kernel name based on the config and hint'''
        pass


class Operator(ABC):

    def __init__(self, name, config: OperatorConfig, target: Target = None):
        if isinstance(target, str):
            target = Target(target)
        self.name = name
        self.config = config
        self.target = target
        self.prim_func_mod = self._select_implementation()
        self.optimized_mod = None
        self.rt_mod = None
        self.time_evaluator = None
        self.arch = get_arch(target) if target else None
        self.dynamic_range = None
        self.pass_context: Dict = {}
        self.num_args = len(self.prim_func.params)
        self.num_output_args: int = (
            1  # todo(lei): should be analyzed from the prim_func.
        )
        self.kernel_name_generator: Optional[BaseKernelNameGenerator] = (
            self.get_kernel_name_generator())
        self.lib_generator = LibraryGenerator(self.arch)
        self.wrapper = TIRWrapper(self.arch)
        self.lib = None

    def get_kernel_name_generator(self) -> Optional[BaseKernelNameGenerator]:
        return None

    def get_source(self, target: Optional[Target] = None, kenrel_only=False) -> str:
        if target is None:
            target = self.target
        if self.lib_generator.lib_code is not None and not kenrel_only:
            return self.lib_generator.lib_code
        if self.rt_mod is None:
            self._build_runtime_module(target)
        return self.rt_mod.imported_modules[0].get_source() if self.rt_mod else None

    def _build_runtime_module(self, target: Target):
        """
        Builds the runtime module based on the architecture platform.

        This function attempts to build a runtime module (rt_mod) for the specified target.
        If the platform is CUDA and an optimized function is available, it tries to build
        using the optimized function with a specific pass context. Otherwise, it falls back
        to building with the primary function. After successful build, it initializes a
        time evaluator for performance measurement.

        Args:
            target (Target): The compilation target specification.

        Returns:
            The compiled runtime module or None if the build was unsuccessful.
        """

        # Initialize rt_mod as None to handle cases where build fails or is skipped
        rt_mod = None

        # Check if the platform is CUDA and we have an optimized function
        if self.arch.platform == "CUDA":
            if self.optimized_mod is None:
                return None

            @tvm.register_func(func_name="tvm_callback_cuda_postproc", override=True)
            def tvm_callback_cuda_postproc(code, _):
                return self.post_process(code)

            try:
                with tvm.transform.PassContext(config={
                        "tir.use_async_copy": True,
                        "tir.disable_cse_tir": True,
                        **self.pass_context
                }):
                    rt_mod = tvm.build(self.optimized_mod, target=target)
            except Exception:  # noqa: F841
                logger.debug(
                    BUILD_RUNTIME_LIBRARY_FAILED_MESSAGE.format(self.__class__.__name__, target,
                                                                "optimized",
                                                                "Failed to build optimized module"))
        else:
            # For non-CUDA platforms or when no optimized function is available, build with the primary function
            rt_mod = tvm.build(self.prim_func, target=target, name=self.name)

        # If the runtime module was successfully built, set up for evaluation
        if rt_mod:
            self.rt_mod = rt_mod
            # Initialize a time evaluator with the built module, specifying the device and the number of runs
            self.time_evaluator = rt_mod.time_evaluator(
                rt_mod.entry_name, self.arch.device, number=10)
            self.torch_func = to_pytorch_func(rt_mod)
            if self.arch.platform == "CUDA":
                try:
                    is_dynamic = (
                        self.dynamic_range is not None and len(self.optimized_mod.functions) > 1)
                    self.wrapper.assign_optimized_module(self.optimized_mod)
                    wrapped_source = self.wrapper.wrap(
                        self.get_source(target, kenrel_only=True), is_dynamic)
                    self.lib_generator.update_lib_code(wrapped_source)
                    self.lib_generator.compile_lib()
                    self.lib = self.lib_generator.load_lib()
                    self.lib.init()

                except Exception as e:
                    build_runtime_library_error = e
                    logger.debug(
                        "Failed to build runtime library {}".format(build_runtime_library_error))

        return rt_mod

    def apply_default_schedule(self, func_mod: IRModule, target: Target) -> IRModule:
        mod_for_opt = deepcopy(func_mod)
        with target:
            optimized_mod = (
                bitblas.ApplyDefaultSchedule(  # pylint: disable=not-callable
                    bitblas.gpu.Matmul(),
                    bitblas.gpu.GEMV(),
                    bitblas.gpu.Reduction(),
                    bitblas.gpu.GeneralReduction(),
                    bitblas.gpu.Fallback(),
                )(mod_for_opt))

        if optimized_mod is not None:
            return optimized_mod
        return None

    def _update_optimized_mod(self, optimized_mod: IRModule):
        self.optimized_mod = optimized_mod

    def _build_default_module(self, target: Target):
        try:
            scheduled_mod = self.apply_default_schedule(self.prim_func_mod, target)
            assert len(scheduled_mod.get_global_vars()) == 1, (
                "The optimized module should only have one global variable for default schedule.")
            assert "main" in scheduled_mod, (
                "The optimized module should have a function named 'main' for default schedule.")
            default_kernal_name = self.kernel_name_generator.generate()
            func = scheduled_mod["main"].with_attr("global_symbol", default_kernal_name)
            optimized_mod = tvm.IRModule({default_kernal_name: func})
            self._update_optimized_mod(optimized_mod)
        except Exception as apply_schedule_error:
            self.optimized_mod = None
            logger.warning(
                APPLY_SCHEDULE_FAILED_MESSAGE.format(self.__class__.__name__, target, "default",
                                                     apply_schedule_error))

        self._build_runtime_module(target)

    def post_process(self, code: str) -> str:
        return code

    def apply_fast_tuning(self,
                          func: PrimFunc,
                          target: Target,
                          topk: int = 20,
                          parallel_build=True) -> Tuple[IRModule, Hint]:
        _, best = fast_tune(func, target, topk=topk, parallel_build=parallel_build)
        # annotate the best pass context
        # TODO(lei): actually we should remove this by enable pass through
        # annotation in the func's attribute.
        self.pass_context = best.config.pass_context
        return ((best.sch.mod, best.config) if best is not None else (None, None))

    def apply_fast_tuning_with_dynamic_range(
        self,
        func: PrimFunc,
        target: Target,
        topk: int = 20,
        dynamic_range: Dict[str, List[int]] = None,
    ):
        optimized_mod = fast_tune_with_dynamic_range(
            func,
            target,
            topk=topk,
            parallel_build=True,
            dynamic_range=dynamic_range,
            kernel_name_generator=self.kernel_name_generator)
        if optimized_mod is not None:
            return optimized_mod
        return None

    def hardware_aware_finetune(self,
                                topk: int = 20,
                                target: Optional[tvm.target.Target] = None,
                                parallel_build=True):
        if target is None:
            target = self.target
        dynamic_range = self.dynamic_range
        func = self.prim_func
        if dynamic_range is not None:
            self.optimized_mod = self.apply_fast_tuning_with_dynamic_range(
                func, target, topk, dynamic_range)
        else:
            scheduled_mod, best_hint = self.apply_fast_tuning(
                func, target, topk, parallel_build=parallel_build)
            assert len(scheduled_mod.get_global_vars()) == 1, (
                "The optimized module should only have one global variable for default schedule.")
            assert "main" in scheduled_mod, (
                "The optimized module should have a function named 'main' for default schedule.")
            default_kernal_name = self.kernel_name_generator.generate(best_hint)
            func = scheduled_mod["main"].with_attr("global_symbol", default_kernal_name)
            optimized_mod = tvm.IRModule({default_kernal_name: func})
            self._update_optimized_mod(optimized_mod)

        self._build_runtime_module(self.target)

    def get_profile_tensors(self, dynamic_symbolic_constraints: Optional[Dict] = None):
        if dynamic_symbolic_constraints is None:
            dynamic_symbolic_constraints = {}
        func = self.prim_func
        device = self.arch.device

        def var_warpper(v):
            if isinstance(v, tvm.tir.Var):
                if v.name in dynamic_symbolic_constraints:
                    return dynamic_symbolic_constraints[v.name]
                assert "opt_shapes" in func.attrs
                assert v.name in func.attrs["opt_shapes"]
                if isinstance(func.attrs["opt_shapes"][v.name], tvm.tir.IntImm):
                    return func.attrs["opt_shapes"][v.name].value
                elif isinstance(func.attrs["opt_shapes"][v.name], tvm.ir.container.Array):
                    avg_shape: int = 0
                    for i in func.attrs["opt_shapes"][v.name]:
                        avg_shape += i.value
                    avg_shape = avg_shape // len(func.attrs["opt_shapes"][v.name])
                    _info_message = f"Doesn't provide dynamic symbolic constrains for {v.name} when do benchmarking, "\
                        f"use average shape {avg_shape}"
                    logger.info(_info_message)
                    return avg_shape
                else:
                    raise RuntimeError("Not supported type: ",
                                       type(func.attrs["opt_shapes"][v.name]))

            elif isinstance(v, tvm.tir.IntImm):
                return v.value
            else:
                raise RuntimeError("Not supported type: ", type(v))

        def map_numpy_type(intype):
            typemap = {
                "e4m3_float8": "float8_e4m3fn",
                "e5m2_float8": "float8_e5m2",
            }
            if intype in typemap:
                return typemap[intype]
            else:
                return intype

        profile_tensors = []
        for param in func.params:
            if param not in func.buffer_map:
                # in case of dynamic symbolic may in params
                continue
            arg = func.buffer_map[param]
            numpy_dtype = map_numpy_type(arg.dtype)
            profile_tensors.append(
                tvm.nd.array(
                    np.random.uniform(0, 1,
                                      [var_warpper(i) for i in arg.shape]).astype(numpy_dtype),
                    device=device,
                ))
        return profile_tensors

    def profile_latency(self, dynamic_symbolic_constraints: Optional[Dict] = None) -> str:
        if dynamic_symbolic_constraints is None:
            dynamic_symbolic_constraints = {}
        profile_tensors = self.get_profile_tensors(dynamic_symbolic_constraints)
        latency = self.time_evaluator(*profile_tensors).mean * 1e3
        # release the memory
        for tensor in profile_tensors:
            del tensor
        return latency

    def _tensor_adapter(self, tensor, device):
        import torch
        from torch.utils.dlpack import to_dlpack

        if isinstance(tensor, tvm.te.Tensor):
            return tensor
        elif isinstance(tensor, torch.Tensor):
            return tvm.runtime.ndarray.from_dlpack(to_dlpack(tensor))
        elif isinstance(tensor, np.ndarray):
            return tvm.nd.array(tensor, device=device)
        else:
            raise RuntimeError("Not supported type: ", type(tensor))

    def _forward_from_torch_func(self, *args):
        # Torch func is not reliable as the runtime overhead dlpack
        # is not negaliable, ref to https://discuss.tvm.apache.org/t/strange-overhead-of-tvm-runtime-ndarray-from-dlpack/16516
        self.torch_func(*args)
        return args[-1]

    def forward(self, *args):
        return self._forward_from_torch_func(*args)

    def _forward_from_prebuild_lib(self, *args, stream=0):
        ctypes_args = [
            ctypes.c_void_p(arr.data_ptr()) if not isinstance(arr, int) else arr for arr in args
        ]
        ctypes_args.append(ctypes.c_void_p(stream))
        self.lib.call(*ctypes_args)

    def call_lib(self, *args, stream=0):
        self.lib.call(*args, ctypes.c_void_p(stream))

    def __call__(self, *args: Any) -> Any:
        return self.forward(*args)

    def update_func(self, func: PrimFunc):
        self.prim_func_mod["main"] = func

    def update_runtime_module(self, rt_mod, srcpath=None, libpath=None):
        self.rt_mod = rt_mod
        self.time_evaluator = rt_mod.time_evaluator(rt_mod.entry_name, self.arch.device, number=10)
        self.torch_func = to_pytorch_func(rt_mod)
        if srcpath is not None:
            assert self.lib_generator is not None, "lib_generator is not initialized"
            self.lib_generator.set_src_path(srcpath)
        if libpath is not None:
            assert self.lib_generator is not None, "lib_generator is not initialized"
            self.lib_generator.set_lib_path(libpath)
            self.lib = ctypes.CDLL(libpath)
            self.lib.init()
        # TODO: update the lib code from srcpath

    def cleanup(self):
        raise NotImplementedError

    @abstractmethod
    def _select_implementation(self) -> IRModule:
        pass

    @property
    def prim_func(self):
        if len(self.prim_func_mod.get_global_vars()) == 1:
            return self.prim_func_mod[self.prim_func_mod.get_global_vars()[0]]
        elif "main" in self.prim_func_mod:
            return self.prim_func_mod["main"]
        else:
            raise ValueError("Unable to determine primary function.")

    @property
    def srcpath(self):
        return self.lib_generator.get_source_path()

    @property
    def libpath(self):
        return self.lib_generator.get_lib_path()

    @property
    def wrapped_source(self):
        return self.lib_generator.lib_code


class OPExecutorCPU:
    """
    A class to execute a sequence of operators on the CPU.
    """

    def __init__(self, operators: Optional[List[Operator]] = None):
        if operators is None:
            operators = []
        self.operators = operators

    def append(self, op):
        self.operators.append(op)

    def is_none(self):
        return len(self.operators) == 0

    def forward(self, weight):
        inputs = [weight]
        for op in self.operators:
            inputs = [op.forward(*inputs)]
        return inputs[-1]

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    @property
    def size(self):
        return len(self.operators)
