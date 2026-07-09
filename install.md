# TVM 源码安装指南（CUDA + cuDNN + CUTLASS + LLVM）

本文档记录在本机从源码编译 Apache TVM 的完整流程，启用 CUDA、cuDNN、CUTLASS 和 LLVM 支持。

源码目录：`/home/zhangxa/codes/edgeLLM/tvm`

---

## 1. 系统依赖

```bash
sudo apt update
sudo apt install -y \
    cmake git ninja-build \
    zlib1g-dev libxml2-dev \
    python3 python3-dev python3-pip \
    llvm-20-dev
```

可选（若需 LLVM 静态链接）：`sudo apt install libpolly-20-dev`

**推荐用 conda 管理 LLVM/CMake：**

```bash
conda create -n tvm-build -c conda-forge \
    "llvmdev>=15" "cmake>=3.24" git python=3.11
conda activate tvm-build
```

还需安装：
- **CUDA Toolkit**（本机：`/usr/local/cuda`，版本 13.2）
- **cuDNN**（本机：系统包 `libcudnn.so.9`，位于 `/usr/lib/x86_64-linux-gnu/`）
- **NVIDIA 驱动**（运行 GPU 时需要；编译阶段只需 NVCC）

---

## 2. 获取源码并初始化 Submodule

```bash
cd /home/zhangxa/codes/edgeLLM/tvm
git submodule update --init --recursive
```

CUTLASS 相关 submodule：
- `3rdparty/cutlass`
- `3rdparty/cutlass_fpA_intB_gemm`
- `3rdparty/libflash_attn`
- `3rdparty/tvm-ffi`

---

## 3. 配置 build/config.cmake

```bash
mkdir -p build
cp cmake/config.cmake build/
```

关键选项（编辑 `build/config.cmake`）：

```cmake
# 后端
set(USE_CUDA ON)
set(USE_LLVM "/usr/bin/llvm-config-20")
set(USE_CUDNN ON)
set(USE_CUTLASS ON)
set(USE_CUBLAS ON)

# 编译优化
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(HIDE_PRIVATE_SYMBOLS ON)
set(USE_CCACHE AUTO)

# GPU 架构：有驱动时用 native 自动检测；无驱动或检测失败时手动指定
# RTX 30xx=86, RTX 40xx=89, H100=90a, B100=100a
set(CMAKE_CUDA_ARCHITECTURES native)
```

> **LLVM 链接说明**
>
> - 推荐：`set(USE_LLVM "/usr/bin/llvm-config-20")`（动态链接，无需额外包）
> - 若需静态链接避免与 PyTorch 符号冲突：
>   ```cmake
>   set(USE_LLVM "/usr/bin/llvm-config-20 --link-static")
>   ```
>   需先安装 `libpolly-20-dev`，否则链接阶段会报 `libPolly.a` 找不到。

---

## 4. 编译

```bash
cd /home/zhangxa/codes/edgeLLM/tvm

# 若 build/ 目录已用 Unix Makefiles 配置过，不要加 -G Ninja
cmake -S . -B build
cmake --build build --parallel
```

**常见编译问题：**

| 错误 | 处理 |
|------|------|
| `generator : Ninja Does not match ... Unix Makefiles` | 去掉 `-G Ninja`，或 `rm -rf build/CMakeCache.txt build/CMakeFiles` 后重配 |
| `没有规则可制作目标 libPolly.a` | 去掉 `--link-static`，或 `sudo apt install libpolly-20-dev` |
| CUTLASS 编译极慢 | 开启 `USE_CCACHE AUTO`，首次编译约 30–60 分钟 |

编译成功后，`build/lib/` 下应有：
- `libtvm_compiler.so`
- `libtvm_runtime.so`
- `libtvm_runtime_cuda.so`
- `libtvm_runtime_extra.so`（cuDNN / CUTLASS）

---

## 5. Python 环境

按 TVM 仓库规范，使用 `PYTHONPATH`，**不要用** `pip install -e .`：

```bash
# 安装 tvm-ffi
cd /home/zhangxa/codes/edgeLLM/tvm/3rdparty/tvm-ffi
pip install .

# 环境变量（建议写入 ~/.bashrc）
export TVM_HOME=/home/zhangxa/codes/edgeLLM/tvm
export TVM_LIBRARY_PATH=$TVM_HOME/build/lib
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH

# CUDA 13.2 下建议用 nvcc 而非默认 NVRTC
export TVM_CUDA_COMPILE_MODE=nvcc

pip install numpy cython pytest
```

---

## 6. 验证

### 6.1 检查加载的库

```bash
python3 -c "
import tvm
for name, lib in tvm.base._LOADED_LIBS.items():
    print(f'{name}: {lib}')
"
```

期望输出指向 `$TVM_HOME/build/lib/` 下的 `.so` 文件。

### 6.2 确认编译选项

```bash
grep -E 'USE_CUDA|USE_CUDNN|USE_CUTLASS|USE_LLVM|USE_CUBLAS' \
  $TVM_HOME/build/TVMBuildOptions.txt
```

期望：

```
USE_CUDA ON
USE_LLVM /usr/bin/llvm-config-20
USE_CUDNN ON
USE_CUBLAS ON
USE_CUTLASS ON
```

> 注意：`tvm.support.libinfo()` 在当前版本是 Python fallback，**不能**用来判断 cuDNN/CUTLASS 是否开启，以 `TVMBuildOptions.txt` 和符号检查为准。

### 6.3 检查设备与后端

```bash
python3 -c "
import tvm
from tvm.testing import env

print('has_llvm():', env.has_llvm())
print('has_cuda():', env.has_cuda())
print('cuda exist: ', tvm.device('cuda', 0).exist)
print('cpu exist:  ', tvm.device('llvm').exist)
print('enabled targets:', tvm.runtime.enabled_targets())
"
```

### 6.4 确认 cuDNN / CUTLASS 已编进库

```bash
nm -D $TVM_LIBRARY_PATH/libtvm_runtime_extra.so | grep -i cudnn | head -3
nm -D $TVM_LIBRARY_PATH/libtvm_runtime_extra.so | grep -i cutlass | head -3
```

有输出即表示对应功能已链接。

### 6.5 LLVM codegen + 运行（官方最小测试）

```bash
cd $TVM_HOME
python3 -m pytest tests/python/all-platform-minimal-test/test_minimal_target_codegen_llvm.py -xvs
```

期望：`PASSED`

### 6.6 CUDA codegen + 运行

新版 TVM 的 CUDA 编译要求 kernel 绑定 GPU 线程（`threadIdx` / `blockIdx`），不能直接用未绑定的 `for` 循环。

```bash
export TVM_CUDA_COMPILE_MODE=nvcc

python3 << 'EOF'
import numpy as np
import tvm
from tvm import te

n = 128
A = te.placeholder((n,), "float32", name="A")
B = te.placeholder((n,), "float32", name="B")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
mod = te.create_prim_func([A, B, C])

# 绑定 GPU 线程
sch = tvm.s_tir.Schedule(mod)
loop = sch.get_loops("C")[0]
bx, tx = sch.split(loop, [None, 128])
sch.bind(tx, "threadIdx.x")
sch.bind(bx, "blockIdx.x")

f = tvm.compile(sch.mod, target="cuda")
print("CUDA compile OK")

dev = tvm.device("cuda", 0)
a = tvm.runtime.tensor(np.random.rand(n).astype("float32"), dev)
b = tvm.runtime.tensor(np.random.rand(n).astype("float32"), dev)
c = tvm.runtime.tensor(np.zeros(n, dtype="float32"), dev)
f(a, b, c)
np.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy(), rtol=1e-5)
print("CUDA run OK")
EOF
```

期望输出：

```
CUDA compile OK
CUDA run OK
```

### 6.7 官方 CUDA 测试（可选）

```bash
export TVM_CUDA_COMPILE_MODE=nvcc
python3 -m pytest tests/python/tirx/test_control_flow.py::test_break_continue1 -xvs
```

---

## 7. 常见问题

| 现象 | 原因 | 处理 |
|------|------|------|
| `Memory verification failed ... Did you forget to bind?` | CUDA kernel 未绑定线程 | 使用 `s_tir.Schedule` 做 `bind(threadIdx.x)` 等 |
| NVRTC 报 `cuda_fp8.hpp` 错误 | CUDA 13.2 与 NVRTC 兼容问题 | `export TVM_CUDA_COMPILE_MODE=nvcc` |
| `nvidia-smi` 失败 | 驱动未加载 | 编译不受影响；GPU 运行时才需要驱动 |
| `has_cudnn()` / `has_cutlass()` 返回 False | `libinfo()` 为 fallback | 以 `TVMBuildOptions.txt` 和 `nm` 符号检查为准 |

---

## 8. 快速参考：每次新开终端

```bash
export TVM_HOME=/home/zhangxa/codes/edgeLLM/tvm
export TVM_LIBRARY_PATH=$TVM_HOME/build/lib
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
export TVM_CUDA_COMPILE_MODE=nvcc
```

---

## 9. 示例：用 Relax nn 前端实现一个简单 LLM Decoder（CUDA）

完整脚本见 [`simple_llm_decoder.py`](./simple_llm_decoder.py)。这是一个 GPT/Llama 风格的小型 decoder：

```
Embedding
-> N x [ RMSNorm -> CausalSelfAttention -> 残差
         RMSNorm -> SiLU-gated FFN       -> 残差 ]
-> RMSNorm
-> LM head (logits)
```

### 9.1 关键点

- **模型定义**：继承 `tvm.relax.frontend.nn.Module`，用 `nn.Embedding` / `nn.Linear` / `nn.RMSNorm` 等层，算子用 `from tvm.relax.frontend.nn import op`。
- **Attention**：用 `op.matmul` + `op.softmax` + 因果 mask（`op.triu(op.full(...))`）手写，避免 `op.scaled_dot_product_attention` 对 float16 的限制，同时保证所有 kernel 都能被 DLight 调度。
- **导出**：`model.export_tvm(spec={...})` 得到 `IRModule` 和参数列表；输入用 `nn.spec.Tensor([1, seq_len], "int32")`。
- **CUDA 编译 pipeline**：

```python
from tvm.s_tir import dlight as dl

dev = tvm.device("cuda", 0)
target = tvm.target.Target.from_device(dev)
with target:
    mod = tvm.ir.transform.Sequential([
        relax.get_pipeline("zero"),          # LegalizeOps + Fuse
        dl.ApplyDefaultSchedule(              # GPU 调度
            dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
            dl.gpu.GeneralReduction(), dl.gpu.Fallback(),
        ),
    ])(mod)

ex = tvm.compile(mod, target=target)
vm = relax.VirtualMachine(ex, dev)
```

- **运行**：权重和输入都要放到 GPU（`tvm.runtime.tensor(np_array, dev)`），参数按 `export_tvm` 返回顺序传入：

```python
logits = vm["forward"](input_ids, *gpu_params).numpy()  # (1, seq_len, vocab_size)
```

### 9.2 运行

```bash
export TVM_HOME=/home/zhangxa/codes/edgeLLM/tvm
export TVM_LIBRARY_PATH=$TVM_HOME/build/lib
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
export TVM_CUDA_COMPILE_MODE=nvcc

cd /home/zhangxa/codes/edgeLLM/tvm_tutorials
python3 simple_llm_decoder.py
```

期望输出末尾：

```
=== Forward run on CUDA ===
logits shape: (1, 16, 128)
predicted next token id: <某个 id>
CUDA decoder run OK
```

### 9.3 说明与后续

- 本示例用**静态 `seq_len` + 随机权重**，目的是打通「定义 → 编译 → CUDA 运行」全流程，不做真实推理质量。
- 编译期会有一条 `topi transform.h ... take Fast mode` 的 Warning，来自 embedding gather（`op.take`），只要输入 token id 在 `[0, vocab_size)` 内即可忽略。
- 进阶方向：
  - 用 `PagedKVCache` + `prefill`/`decode` 分离方法实现真正的自回归生成，参考 `$TVM_HOME/docs/how_to/tutorials/optimize_llm.py`。
  - 用动态 `seq_len`（`nn.spec.Tensor([1, "seq_len"], "int32")`）支持变长输入。
  - 切换到 float16 + `op.scaled_dot_product_attention` 或 CUTLASS/cuDNN BYOC 后端提升性能。
