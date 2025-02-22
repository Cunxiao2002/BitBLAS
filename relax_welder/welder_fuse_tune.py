import numpy as np
import os
from typing import Dict
import time
import bitblas
from bitblas import tvm as tvm
from tvm import relay, relax, runtime, transform, ir
from tvm.relax.testing import relay_translator, nn
from tvm.target.target import Target
import tvm.relay.testing
from tvm.ir.module import IRModule
from bitblas.relax import ApplyDefaultSchedule, ApplyFastTuning
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R
# from bitblas.base.welder.test_relax import OperatorExtractor
from bitblas.base import fast_tune
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


@I.ir_module
class MatmulReluModule:
    @R.function
    def main(
        x: R.Tensor((128, 128), "float32"),
        w: R.Tensor((128, 128), "float32")
    ) -> R.Tensor((128, 128), "float32"):
        with R.dataflow():
            # Matrix multiplication
            lv0 = R.matmul(x, w)
            # ReLU activation
            lv1 = R.nn.relu(lv0)
            # Mark output
            R.output(lv1)
        return lv1

# 创建一个将op_pattern统一的pass
@ir.transform.module_pass(opt_level=0)
class CustomAnnotateTIROpPattern:
    def transform_module(self, mod, ctx):
        @tvm.tir.transform.prim_func_pass(opt_level=0)
        def transform(func, mod, ctx):
            return func.with_attr("op_pattern", 0)  # 将所有 TIR 标记为 kElemWise
        return transform(mod)

with transform.PassContext():
    relax_mod = MatmulReluModule
    relax_mod = relax.transform.LegalizeOps()(relax_mod)
    relax_mod = CustomAnnotateTIROpPattern()(relax_mod) # 该pass是有用的
    write_mod(relax_mod, log_path, "CusomAnnotateTIROpPattern")
    relax_mod = relax.transform.FuseOps()(relax_mod)
    write_mod(relax_mod, log_path, "FuseOps")
    relax_mod = relax.transform.FuseTIR()(relax_mod)
    write_mod(relax_mod, log_path, "FuseTIR")

target = tvm.target.Target("cuda")
relax_mod = ApplyFastTuning(topk=20, target=target, parallel_build=True)(relax_mod)
# write_code(relax_mod, log_path, "apply_fast_tuning")
write_mod(relax_mod, log_path, "apply_fast_tuning")

# target = tvm.target.Target("cuda")
# extractor = OperatorExtractor(relax_mod, target=target)
# extractor.visit_expr(relax_mod["main"])

# ordered_nodes = extractor.ordered_nodes
# nodeMap = extractor.node_map
# _, best = fast_tune(ordered_nodes[0].prim_func, target=target)

# print(f"Best latency is {best.latency}")
# print("---------------------------------------------")
# print(f"Best is {best}")