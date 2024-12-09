# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import numpy as np
import os
from typing import Dict
import time
import bitblas
from bitblas import tvm as tvm
from tvm import relay, relax, runtime, transform
from tvm.relax.testing import relay_translator, nn
from tvm.target.target import Target
import tvm.relay.testing
from tvm.ir.module import IRModule
from bitblas.relax import ApplyDefaultSchedule, ApplyFastTuning

fname = os.path.basename(__file__)
fname = os.path.splitext(fname)[0]
# get current file path
log_path = os.path.dirname(os.path.abspath(__file__)) + "/progress/" + fname

count = 0

bitblas.set_log_level("Debug")


def write_code(code, path, fname):
    global count
    fname = str(count) + "." + fname
    count += 1
    if not os.path.exists(path):
        os.makedirs(path)
    fname = os.path.join(path, fname)
    with open(fname, "w") as f:
        f.write(code)


def write_sch(sch, path, fname):
    py_fname = fname + ".py"
    write_code(sch.mod["main"].script(), path, py_fname)
    cu_fname = fname + ".cu"
    write_code(sch.mod.astext(), path, cu_fname)


def write_mod(mod, path, fname):
    py_fname = fname + ".py"
    write_code(mod.script(show_meta=False), path, py_fname)
    cu_fname = fname + ".cu"
    write_code(mod.astext(show_meta_data=False), path, cu_fname)

from tvm.relay.frontend.onnx import from_onnx
import onnx

model_path = "/root/BitBLAS/examples/NeRF-b128/model.onnx"
onnx_model = onnx.load(model_path)
relay_mod, params = from_onnx(onnx_model)

target = tvm.target.Target("cuda")
dtype = "float32"

def apply_opt_before_tuning(relay_mod: IRModule, params: Dict[str, runtime.NDArray], target: Target):
    with transform.PassContext(opt_level=3):
        main_func = relay_mod["main"]
        bind_main_func = relay.build_module.bind_params_by_name(main_func, params)
        relay_mod = IRModule.from_expr(bind_main_func)
        write_mod(relay_mod, log_path, "create_mod")
        relay_mod = relay.transform.SimplifyInference()(relay_mod)
        write_mod(relay_mod, log_path, "SimplifyInference")
        relay_mod = relay.transform.ConvertLayout({"nn.conv2d": ["NHWC", "default"]})(relay_mod)
        write_mod(relay_mod, log_path, "ConvertLayout")
        relay_mod = relay.transform.FoldConstant()(relay_mod)
        write_mod(relay_mod, log_path, "FoldConstant")
        relay_mod = relay.transform.FoldScaleAxis()(relay_mod)
        write_mod(relay_mod, log_path, "FoldScaleAxis")
        relay_mod = relay.transform.CanonicalizeOps()(relay_mod)
        write_mod(relay_mod, log_path, "CanonicalizeOps")
        relay_mod = relay.transform.AlterOpLayout()(relay_mod)
        write_mod(relay_mod, log_path, "AlterOpLayout")
        relay_mod = relay.transform.FoldConstant()(relay_mod)
        write_mod(relay_mod, log_path, "FoldConstant")

        # opt_level=2 and select_impl_strategy are required for avoiding winograd lowering
        relax_mod = relay_translator.from_relay(
            relay_mod["main"],
            opt_level=2,
            target=target,
            append_op_attrs=True,
            select_impl_strategy="first")
        write_mod(relax_mod, log_path, "relay_translator_relax")
        relax_mod = relax.transform.AnnotateTIROpPattern()(relax_mod)
        write_mod(relax_mod, log_path, "AnnotateTIROpPattern")
        relax_mod = relax.transform.FuseOps()(relax_mod)
        write_mod(relax_mod, log_path, "FuseOps")
        relax_mod = relax.transform.FuseTIR()(relax_mod)
        write_mod(relax_mod, log_path, "FuseTIR")
    return relax_mod

relax_mod = apply_opt_before_tuning(relay_mod, params, target)

#print(relax_mod)

start_tune_time = time.time()
relax_mod = ApplyFastTuning(topk=20, target=target, parallel_build=False)(relax_mod)
end_tune_time = time.time()

write_code(relax_mod, log_path, "apply_opt_before_tuning")