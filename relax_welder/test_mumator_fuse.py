from tvm import tir
from tvm.ir.attrs import DictAttrs, make_node
from tvm.relax.expr_functor import PyExprMutator
import tvm
from tvm.script import tir as T
import bitblas

def analyze_operator(op_func: tir.PrimFunc) -> dict:
    """分析算子"""
    analysis = {
        "inputs": [],
        "params": [],
        "output": None,
        "block": None  # 存储主计算块
    }

    # 分析参数和缓冲区
    for param in op_func.params:
        buffer = op_func.buffer_map[param]
        if param == op_func.params[-1]:
            analysis["output"] = (param, buffer)
        elif "param" in param.name:
            analysis["params"].append((param, buffer))
        else:
            analysis["inputs"].append((param, buffer))

    # 获取主计算块
    def visit(stmt):
        if isinstance(stmt, tir.stmt.Block):
            analysis["block"] = stmt
    tir.stmt_functor.post_order_visit(op_func.body, visit)
    return analysis

@tvm.relax.expr_functor.mutator
class FusionMutator(PyExprMutator):
    def __init__(self, op1: tir.PrimFunc, op2: tir.PrimFunc):
        super().__init__()
        self.op1 = op1
        self.op2 = op2
        self.op1_analysis = analyze_operator(op1)
        self.op2_analysis = analyze_operator(op2)

    def create_fused_params(self):
        """创建融合后的参数和buffer映射"""
        fused_params = []
        fused_buffer_map = {}

        # 添加输入参数
        for param, buffer in self.op1_analysis["inputs"]:
            fused_params.append(param)
            fused_buffer_map[param] = buffer

        # 添加权重等参数
        for param, buffer in self.op1_analysis["params"] + self.op2_analysis["params"]:
            fused_params.append(param)
            fused_buffer_map[param] = buffer

        # 添加输出参数
        output_param, output_buffer = self.op2_analysis["output"]
        fused_params.append(output_param)
        fused_buffer_map[output_param] = output_buffer

        return fused_params, fused_buffer_map

    def visit_primfunc(self, func: tir.PrimFunc) -> tir.PrimFunc:
        """访问并转换PrimFunc，在block层面融合两个kernel，并共用相同的blockIdx和共享内存"""
        # 声明共享内存，scope设置为'shared'
        shared_buffer = tir.decl_buffer(
            self.op1_analysis["output"][1].shape,
            self.op1_analysis["output"][1].dtype,
            "shared_memory",
            scope="shared",
        )
        # 构造融合后的block，并使用op1中的iter_vars来共享blockIdx
        fused_block = tir.Block(
            iter_vars=self.op1_analysis["block"].iter_vars,
            reads=self.op1_analysis["block"].reads,
            writes=list(self.op1_analysis["block"].writes) + list(self.op2_analysis["block"].writes),
            name_hint="fused_" + self.op1_analysis["block"].name_hint + "_" + self.op2_analysis["block"].name_hint,
            body=tir.SeqStmt([
                self.op1_analysis["block"].body,
                self.visit_stmt(self.op2_analysis["block"].body)
            ])
        )
        fused_params, fused_buffer_map = self.create_fused_params()
        fused_name = f"fused_{self.op1.attrs['global_symbol']}_{self.op2.attrs['global_symbol']}"
        return tir.PrimFunc(
            params=fused_params,
            buffer_map=fused_buffer_map,
            body=fused_block,
            attrs=tvm.ir.make_node("DictAttrs", global_symbol=fused_name)
        )

    def visit_stmt(self, stmt: tir.Stmt) -> tir.Stmt:
        """
        访问并转换语句：如果加载的buffer为op2中第一个输入，则将其替换成共享内存，
        以实现两个kernel公用同一块共享内存。
        """
        if isinstance(stmt, tir.BufferLoad):
            if stmt.buffer == self.op2_analysis["inputs"][0][0]:
                shared_buffer_inst = tir.decl_buffer(
                    self.op1_analysis["output"][1].shape,
                    self.op1_analysis["output"][1].dtype,
                    scope="shared",
                )
                return tir.BufferLoad(shared_buffer_inst, stmt.indices)
        # 对于其他语句，直接递归转换
        return stmt

def fuse_operators(op1: tir.PrimFunc, op2: tir.PrimFunc) -> tir.PrimFunc:
    """使用Mutator机制融合两个算子"""
    mutator = FusionMutator(op1, op2)
    return mutator.visit_primfunc(op1)

def create_gemm(M, N, K):
    @T.prim_func
    def gemm(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, [M, K], dtype="float16")
        B = T.match_buffer(b, [K, N], dtype="float16")
        C = T.match_buffer(c, [M, N], dtype="float16")

        for i, j, k in T.grid(M, N, K):
            with T.block("gemm"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float16(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    return gemm

if __name__ == "__main__":
    gemm1 = create_gemm(512, 512, 128)
    gemm2 = create_gemm(512, 128, 512)

    # 调用 fuse_operators 对两个 GEMM 算子进行融合
    fused_func = fuse_operators(gemm1, gemm2)
    print("fused PrimFunc:")
    print(fused_func)