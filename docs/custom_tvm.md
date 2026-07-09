# 可定制 IR 编译框架 vs 黑盒推理引擎

除了 TVM 这种可以在 IR 上自由做各种定制优化的框架外，当前生态里还有大量其他方案。本文按**「IR / 优化的开放与可定制程度」**这条主轴，梳理主流技术路线，便于在 edge LLM 场景下做架构选型。

**本文「端侧」的含义**：不是手机 / MCU，而是**机器人边缘算力板**，典型目标为：

| 平台 | 算力形态 | 官方主栈倾向 |
|------|----------|--------------|
| **NVIDIA Jetson（含 Thor 等）** | CUDA GPU + CPU | TensorRT / **TensorRT Edge-LLM**、CUDA、部分 PyTorch |
| **地平线（Horizon）等机器人芯片** | BPU / NPU + CPU | 厂商工具链（如 OpenExplorer）、量化与图编译偏黑盒 |

因此选型要同时回答两件事：**(1) 单平台 peak 性能**（常走 vendor 黑盒）；**(2) 跨 Jetson ↔ 地平线 的可移植与可定制 IR**（更适合 TVM / MLIR 一类）。

---

## 1. 光谱视角

```
全黑盒 ←──────────────────────────────────────────→ 全开放 / 可编程 IR

cuDNN / TensorRT   OpenVINO / ORT   XLA / OpenXLA   TorchInductor   TVM / MLIR / IREE / Hidet   Triton / CUTLASS
(算子 / 引擎级)     (可插 EP)         (HLO 中层)       (FX + Inductor)  (多层 IR 自由改写)          (kernel 级手写)
```

**TVM 的位置**：Relax / TIR 多层 IR + 可编程 schedule + BYOC（如 CUTLASS / cuDNN），属于光谱最右侧「可自由定制 IR」这一档。

**TensorRT 的位置**：核心融合策略、kernel 选择、内存规划多为黑盒，用户主要通过 builder 配置和 plugin 扩展，定制空间相对有限。

---

## 2. 与 TVM 同类：开放 IR、可定制编译器

| 框架 | IR 层次 | 定制方式 | 生态定位 |
|------|---------|----------|----------|
| **MLIR + IREE** | 多层 dialect（linalg / affine / vector / gpu …） | 自定义 dialect、pass、lowering，灵活度极高 | 工业界最活跃的可定制编译基础设施之一，Google 主导，端到端多后端（CPU / GPU / Vulkan / SPIR-V） |
| **OpenXLA / XLA + StableHLO** | HLO / StableHLO | 可写 pass、custom call，但比 TVM 更封闭 | JAX / TensorFlow 默认编译栈，TPU 一等公民，编译式 fusion |
| **PyTorch TorchInductor**（`torch.compile`） | FX graph → Inductor IR → Triton | 写 lowering、custom pass、自定义后端 | PyTorch 2.x 原生路径，落地最快；**本机 JIT，非跨芯片部署主路径**（详见 §2.1） |
| **Hidet** | 任务-调度式 IR（思路接近 TVM） | Python 可编程调度 | 已成为 PyTorch 官方后端之一，图级 + 算子级编译 |
| **Halide / Tiramisu / TACO** | compute / schedule 分离 | 手写 schedule | Halide 是 TVM 的思想源头；Tiramisu 走多面体；TACO 专攻稀疏张量 |

**要点**：若目标是「在 IR 上自由改写、做研究型或深度定制优化」，**MLIR / IREE** 是与 TVM 最直接对标的工业级方案；**TorchInductor** 是「生态最大、接入成本最低」的本机加速路径——对 **Jetson** 有限可用，对 **地平线 BPU** 基本不对口，见下。

### 2.1 TorchInductor：解决什么问题？对机器人端侧怎么样？

`torch.compile` → TorchInductor 的核心目标是：

1. **减少 Python / dispatcher 开销**：把 eager 执行变成图级编译。
2. **算子融合与代码生成**：FX 图 → Inductor IR → **Triton（CUDA GPU）** 或 **C++ / OpenMP（CPU）**。
3. **尽量不改用户代码**：`torch.compile(model)` 即可加速。
4. **本机加速**：面向「已经在跑 PyTorch 的那台机器」（云 GPU、工作站、部分 Jetson），**不是**「一套 IR 编译到多种机器人芯片」。

一句话：**本机 JIT，不是跨平台 AOT 部署器。**

针对本文定义的端侧（Jetson Thor / 地平线）：

| 维度 | Jetson（Thor 等） | 地平线等 BPU / NPU |
|------|-------------------|---------------------|
| **能否跑 Inductor** | 有 CUDA + 能装 PyTorch 时，**可以尝试**本机 `torch.compile` | **基本不行**：算力在 BPU，Inductor 不会生成 BPU 图 |
| **是否官方主路径** | 否；量产 LLM 更常走 **TensorRT Edge-LLM**（非桌面 TensorRT-LLM） | 否；走 **厂商工具链**（量化、图编译、runtime） |
| **Triton 依赖** | Jetson 上 Triton / Inductor 成熟度通常弱于桌面 A100/H100，需实测 | 无关 |
| **跨平台** | 仅覆盖 NVIDIA CUDA 系 | 不覆盖 |
| **与 TVM 关系** | 可作 Jetson 上的 PyTorch 原型加速；量产 / 跨芯片仍看 TVM 或 TRT | 跨芯片统一 IR 更依赖 **TVM / 自研 lowering**，而非 Inductor |

**结论（按平台拆开）**：

- **Jetson**：TorchInductor **不是不能用**，而是定位是「板子上还想用 PyTorch 时的加速器」。要 peak 性能与工程交付，优先 **TensorRT Edge-LLM**；要可定制 IR、与地平线共用一套编译思路，优先 **TVM（CUDA 后端）**。
- **地平线**：TorchInductor **几乎不在候选名单**。要么厂商栈，要么 TVM / 自研把图 lower 到 BPU（工作量大，但可控）。
- **同时覆盖 Jetson + 地平线**：Inductor 无法当统一方案；**TVM / MLIR 一类开放编译器**才是「一套前端、多后端」的合理底座；各平台再用 vendor 库做单点加速（Jetson 上 BYOC / 对照 Edge-LLM，地平线上接厂商 runtime）。

与机器人端侧相关路线的分工：

```
TorchInductor：     PyTorch 图 → 本机更快（主要 CUDA/CPU；Jetson 上可选）
TensorRT Edge-LLM： Jetson 上 LLM/VLM peak（黑盒偏多；非桌面 TensorRT-LLM）
厂商工具链：        地平线 BPU 上的官方部署路径（黑盒偏多）
TVM / MLC / MLIR：  可定制 IR + 跨 Jetson/CPU/（潜在）NPU 后端
ExecuTorch：        更偏手机 / 轻量嵌入式，不是机器人 SoC 主战场
```

| | TorchInductor | TensorRT Edge-LLM | TVM / MLC | 地平线工具链 |
|--|---------------|-------------------|-----------|--------------|
| 问题域 | 本机 PyTorch 加速 | Jetson 边缘 LLM 极致推理 | 可定制编译 / 跨后端 | BPU 官方部署 |
| Jetson | 可用（原型） | **强** | 强（CUDA） | — |
| 地平线 | 弱 / 无 | 无 | 需自研或社区后端 | **强** |
| IR 可定制 | 中 | 低 | **高** | 低 |
| 跨两平台统一 | 否 | 否 | **最合适** | 否 |

对 **edgeLLM（Jetson Thor + 地平线）**：统一可定制底座继续押 **TVM**；Jetson 量产可并行用 **TensorRT Edge-LLM** 做性能天花板；地平线走 **厂商栈或 TVM→BPU**。Inductor 最多作为 Jetson 上的 PyTorch 研发加速手段，**不要当成跨芯片主栈**。

---

## 3. Kernel 级 DSL：比整图 IR 更细粒度的定制

这些通常不做完整图编译，而是让你**手写或半自动生成高性能 kernel**，常与上层编译器组合使用：

| 技术 | 说明 |
|------|------|
| **Triton**（OpenAI） | Python 写 GPU kernel，自动做 tiling / 流水线；TorchInductor 的代码生成后端之一。介于「手写 CUDA」与「全自动编译」之间，可定制性很高 |
| **CUTLASS / CuTe**（NVIDIA） | C++ 模板库，GEMM / 卷积的可组合构建块。TVM 可通过 `USE_CUTLASS` 作为 BYOC 后端接入 |
| **FlashAttention / ThunderKittens** | 面向 Attention 等特定算子的高性能 kernel 库 |
| **Mojo**（Modular） | 新语言，试图统一 Python 易用性与系统级性能、可编程 kernel |

**与 TVM 的关系**：TVM 负责整图 lowering 与调度；Triton / CUTLASS 负责单算子极限性能。二者可互补，而非互斥。

---

## 4. 半开放：核心黑盒，但可插件扩展

| 框架 | 开放性 | 扩展方式 |
|------|--------|----------|
| **ONNX Runtime** | 图优化偏黑盒 | **Execution Provider（EP）**：可插入 TensorRT、OpenVINO、CUDA、自定义后端 |
| **OpenVINO**（Intel） | Intel 硬件推理引擎，优化策略不可见 | 自定义算子、部分图改写 |
| **TensorRT** | 融合 / kernel 选择黑盒 | **Custom Plugin** 补自定义算子；Builder 配置有限 |

这类方案的共性：**主干优化 pipeline 不可改，扩展点集中在「算子插件」或「后端 EP」**。

---

## 5. 黑盒 Vendor 库（与 TensorRT 同类）

- **NVIDIA**：cuDNN、cuBLAS、TensorRT；桌面/服务器另有 TensorRT-LLM；**Jetson Thor 边缘 LLM 对齐 TensorRT Edge-LLM**
- **Intel**：oneDNN（DNNL）
- **Apple**：CoreML
- **华为**：Ascend CANN

**优点**：开箱即用，单点性能往往极强，与自家硬件深度绑定。

**缺点**：融合策略、kernel 选择、内存规划不可见、不可改；跨平台与长期可维护性依赖 vendor 路线图。

---

## 6. LLM 专用推理引擎（edge LLM 重点）

这些不是通用编译器，而是「**运行时 + 高度优化 kernel + 调度策略**」的组合：

| 引擎 | 开放性 | 特点 |
|------|--------|------|
| **MLC-LLM** | 开放（**基于 TVM Unity / Relax**） | 与 TVM 技术栈同源；跨平台（手机 / 边缘 / 浏览器 WebGPU）；可编译定制 |
| **TensorRT Edge-LLM** | 黑盒偏多 | **Jetson** 上 LLM/VLM 官方边缘路径；桌面/服务器对应 TensorRT-LLM |
| **llama.cpp / ggml** | 开放 | 端侧 / CPU 首选，手写 kernel，量化生态强 |
| **vLLM / SGLang** | 开放（Python + Triton / CUDA） | 服务端高吞吐，PagedAttention / RadixAttention |
| **LMDeploy / TGI** | 半开放 | 服务化部署，偏工程落地 |

---

## 7. 选型维度对比

| 维度 | TVM / MLIR / IREE | TorchInductor + Triton | TensorRT Edge-LLM（Jetson） |
|------|-------------------|------------------------|-------------------------|
| IR 可见性 | 高 | 中（FX / Inductor 可见） | 低 |
| 自定义 pass / schedule | 是 | 部分（custom pass、Triton） | 否（仅 plugin） |
| Jetson | 强（CUDA 后端） | 可用（本机 PyTorch 加速） | **最强（官方为 TensorRT Edge-LLM，见 §8）** |
| 地平线 BPU | 需自研 / 社区后端，但架构上可扩展 | **基本不可用** | 无 |
| 跨 Jetson+地平线 | **最合适的统一底座** | 否 | 否 |
| 接入成本 | 高 | 低（`torch.compile`） | 中（工程化 TRT / Edge-LLM 流程） |
| 单点 peak | 取决于优化投入 | 中高（Jetson 上需实测） | 往往最高（同 vendor） |

---

## 8. Jetson Thor：Inductor vs TVM vs TensorRT Edge-LLM

本节专门回答：**在 Jetson Thor 上走 PyTorch Inductor 如何？** 从延迟、内存、成本与 TVM / 官方路径对比。

### 8.1 Thor 上实际有哪些路

| 路径 | 形态 | Thor 上的现实 |
|------|------|----------------|
| **TensorRT Edge-LLM** | C++ runtime + TensorRT engine | NVIDIA 官方边缘 LLM / VLM 路径。注意：标准 **TensorRT-LLM 不在 Jetson 上提供**（论坛已明确）；Thor 应对齐 **TensorRT Edge-LLM**（JetPack 7.x） |
| **PyTorch + `torch.compile`（Inductor）** | 板上仍跑 PyTorch，JIT 出 Triton / CUDA | 可行前提：有匹配 **CUDA 13 / sm_110（Blackwell）** 的 PyTorch；Triton 在 Jetson 上的成熟度需实测 |
| **TVM（CUDA）** | AOT 编译成 runtime 模块 | 可定制 IR；LLM 需自建或接 MLC 类栈；单点 peak 通常不如 Edge-LLM |
| **vLLM 等** | 服务式吞吐 | Thor 上有探索，偏吞吐，不是机器人硬实时主路径 |

Thor 硬件量级（以 T5000 为例）：约 **128GB 统一内存**、**273 GB/s**、Blackwell GPU、**40–130W**。内存比 Orin 宽裕，但 LLM decode 仍常受 **带宽** 约束，不是「算力不够」那么简单。

### 8.2 最终性能 / 延迟

**Inductor 擅长**：融合 elementwise / 小算子，减少 kernel launch 与中间写回；对「标准 Transformer + 仍要 Python 调试」友好。相对 **eager PyTorch**，常见大约 **1.2×–2×**（视模型与是否 `max-autotune` 而定）。

**Inductor 在 Thor 上的短板**（相对 Edge-LLM / 精调 TVM）：

1. LLM decode 关键不在「再融几个 pointwise」，而在 Attention、GEMM、KV cache、量化（FP8 / NVFP4）、调度与 CUDA Graph——这些正是 **TensorRT Edge-LLM** 针对边缘打磨的点。
2. **Triton 在 Jetson / ARM / sm_110** 上往往不如桌面 A100/H100 稳：autotune 慢、部分 kernel 回退、首次编译冷启动长。
3. **仍带 PyTorch 运行时**：graph break、Python 边界、动态控制流会吃掉实时性余量。
4. 统一内存 + **273 GB/s** 带宽下，decode 多半 memory-bound；若不做激进量化 / Attention 优化，延迟会明显落后官方 FP8 / NVFP4 路径。

相对关系（同模型、同精度量级下的示意，越左延迟越低）：

```
TensorRT Edge-LLM (FP8/NVFP4)  <<  精调 TVM/MLC  <<  Inductor  <<  eager PyTorch
```

| 场景 | Inductor | TVM（认真优化） | TensorRT Edge-LLM |
|------|----------|-----------------|-------------------|
| Prefill | 中 | 中～好 | **通常最好** |
| Decode token 延迟 | 中偏弱 | 中～好（看 KV / 量化） | **通常最好** |
| 硬实时 / 可控尾延迟 | 弱（JIT / Python） | 较好（AOT） | **最好（C++ runtime）** |
| 首包 / 冷启动 | 差（编译 + autotune） | 好（可离线编译） | 好（engine 预构建） |

**延迟结论**：机器人实时 LLM / VLM 场景，Inductor 很难打过 Edge-LLM；相对未深度优化的 TVM，Inductor 可能「更快上手、性能差不多或略好」；相对深度优化的 TVM / MLC，Inductor 通常仍弱在 Attention / 量化 / 部署形态。

### 8.3 内存资源消耗

Jetson 是 **CPU / GPU 统一内存**，要看进程总占用，不是「显存单独一块」。

| 项目 | Inductor | TVM runtime | TensorRT Edge-LLM |
|------|----------|-------------|-------------------|
| **框架常驻** | PyTorch + CUDA + 可能 Triton：**大**（数 GB 级常见） | `libtvm*` 等：**小得多** | 轻量 C++ + TRT：**小** |
| **权重** | 易停在 FP16 / BF16 | 可自定义量化布局 | **FP8 / NVFP4 / INT4 AWQ** 等官方路径，更省 |
| **KV cache** | 一般自管或依赖上层 | 可接 PagedKV 等，可控 | 边缘场景专门优化 |
| **编译期峰值** | `max-autotune` / 追踪时峰值高 | 编译多在主机，板上主要推理 | engine 可在主机构建，板上加载 |
| **128GB Thor** | 「装得下」≠「跑得省」；与感知 / 规划多进程共存时最易挤占 | 更适合多进程共存 | 最适合量产多进程 |

**内存结论**：Thor 内存大，Inductor 不容易 OOM，但 **单位 token 效率与常驻开销** 通常明显差于 Edge-LLM，也往往差于精简 TVM runtime。

### 8.4 成本（工程 / 时间 / 功耗 / 人力）

| 成本项 | Inductor | TVM | TensorRT Edge-LLM |
|--------|----------|-----|-------------------|
| **接入成本** | **最低**（有 PyTorch 就能试） | 高（编译栈、schedule、导出） | 中（量化 / engine 流程） |
| **人力** | 少；调优天花板也低 | 高；换可控性与跨平台 | 中；跟 NVIDIA / JetPack 文档 |
| **编译 / 迭代** | 板上 JIT，迭代快；autotune 耗电耗时 | 主机 AOT，板上只跑 | 主机建 engine，板上稳定 |
| **功耗** | 同等精度下通常更高（效率低 → GPU 占用更久） | 中 | **通常最好** |
| **长期维护** | 绑 PyTorch / Triton / JetPack | 自控强，需养编译能力 | 跟 JetPack / Edge-LLM 版本 |
| **跨地平线** | **几乎无复用** | **唯一较现实的统一 IR** | 无 |

**成本结论**：短期 Demo 用 Inductor 最便宜；量产延迟 / 功耗 / 多进程用 Edge-LLM 综合往往更省；还要地平线时，TVM 的前期投入会在跨平台上摊薄。

### 8.5 怎么选（只谈 Thor）

```
只做 Jetson Thor、要峰值与实时
  → TensorRT Edge-LLM 主路径
  → TVM 做差异化算子 / 研究 / 兜底
  → Inductor 最多做算法原型

要 Jetson + 地平线统一
  → TVM（或 MLIR）做统一前端
  → Thor 上可对照 / BYOC Edge-LLM
  → Inductor 不参与跨芯片

只要最快验证 PyTorch 模型在 Thor 上能不能跑
  → Inductor / eager 都可以
```

| 维度 | Thor 上的赢家 |
|------|----------------|
| Token 延迟 / 吞吐 peak | **TensorRT Edge-LLM** |
| 内存效率（量化 + runtime） | **TensorRT Edge-LLM** |
| 可定制 IR / 新算子 | **TVM** |
| 跨地平线 | **TVM** |
| 研发速度（PyTorch 原样） | **Inductor** |
| 量产确定性（尾延迟、冷启动） | **Edge-LLM > TVM > Inductor** |

**一句话**：Thor 上走 Inductor **可以，但定位是研发 / 原型加速器，不是量产主引擎**。量产对照用 Edge-LLM；研究与跨地平线用 TVM。

---

## 9. 结合 edge LLM 场景的建议（Jetson Thor + 地平线）

若已在用 TVM，目标是机器人芯片而非手机：

1. **统一可定制底座继续押 TVM**：一套 Relax / TIR 前端，Jetson 走 CUDA（+ CUTLASS / cuDNN BYOC）；地平线侧评估「TVM → BPU」或「TVM 管图优化 + 厂商 runtime 执行」的混合方案。
2. **Jetson 量产性能天花板**：**TensorRT Edge-LLM**（不是桌面版 TensorRT-LLM）做 baseline 与交付备选；接受黑盒换 peak，与 TVM 路径并行，而不是二选一。细节对比见 **§8**。
3. **TorchInductor**：仅作 **Jetson / 桌面上仍跑 PyTorch 时的加速手段**；不要指望它覆盖地平线，也不要当跨芯片部署方案。
4. **地平线**：优先摸清厂商工具链能力边界（算子覆盖、量化、LLM 支持）；缺口再用 TVM / 自研补——这是「开放 IR」真正有价值的地方。
5. **MLIR / IREE**：若团队要长期做多 NPU dialect，可作为与 TVM 并列的基础设施选项；短期落地仍以 TVM + vendor 组合更现实。
6. **kernel 级**：**CUTLASS / Triton** 主要服务 Jetson / NVIDIA；地平线侧对应的是厂商 kernel 库，而非 Triton。

推荐组合（示意）：

```
研发 / 可定制：       TVM（Jetson CUDA） ←→ 同一套模型 IR ←→ 地平线（厂商或 TVM-BPU）
Jetson 性能冲刺：     TensorRT Edge-LLM（黑盒）
Jetson PyTorch 原型： torch.compile / Inductor（可选，见 §8）
```

---

## 10. 一句话总结

- **「可自由定制 IR」**：TVM、MLIR / IREE、Hidet、XLA；TorchInductor 属于「本机可定制加速」，不是跨机器人芯片部署器。
- **「kernel 级手写 / 半自动生成」**：Triton、CUTLASS（偏 NVIDIA / Jetson）。
- **「黑盒换 peak 性能」**：TensorRT Edge-LLM（Jetson Thor）、地平线厂商工具链、cuDNN 等。

开放编译器换来的是**可控性、跨 Jetson↔地平线 的可移植性、研究自由度**；黑盒引擎换来的是**单平台极致性能与低接入成本**。机器人 edge LLM 通常需要 **TVM（统一）+ TensorRT Edge-LLM（Jetson peak）+ 厂商栈（地平线）** 的组合，而非非此即彼。

---

## 11. 延伸阅读

- TVM 官方 LLM 优化教程：`$TVM_HOME/docs/how_to/tutorials/optimize_llm.py`
- 本仓库示例：[`simple_llm_decoder.py`](../simple_llm_decoder.py)（Relax nn 前端 + CUDA 编译运行）
- 安装与验证：[`install.md`](../install.md)
- Jetson AI Lab：TensorRT Edge-LLM on Jetson（Thor / Orin 官方边缘 LLM 路径）
- NVIDIA 论坛：Jetson 上使用标准 TensorRT-LLM 的限制说明（应改用 Edge-LLM）
