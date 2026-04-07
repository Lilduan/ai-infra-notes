# Day 2

Date: 2026-03-18
Topic: PyTorch execution from a systems angle
Time spent: 60-90 min target

## Goal

从 **end-to-end critical path** 视角理解 PyTorch execution：区分哪些是 Python overhead，哪些是 backend execution，并开始把一次 forward pass 拆成 compute、memory movement、synchronization 三类成本。

## Today Checklist

- [x] Review `tensor layout`、`autograd`、`operator dispatch`、`eager execution`
- [x] Trace 一个最小 model 的 forward pass
- [ ] 标出 compute / memory movement / synchronization 可能发生的位置
- [ ] 写清 Python overhead 和 backend execution 的边界
- [ ] 记录 5 个关于 PyTorch internals 还答不清的问题

## Suggested Timebox

- 20 min: 补齐 PyTorch execution 主干概念
- 25 min: 跑最小 forward / profiler
- 20 min: 写 execution path breakdown
- 15 min: 整理 open questions

## Output

- Main note: `notes/day02-pytorch-execution.md`
- Optional helper: one small trace or profiler table



---

## 学习记录

一些概念：

- ### tensor layout
  - 类比 C 里的多维数组视图，有一些基于tensor的基本概念：
    - shape：每一维有多大
    - stride：沿着某一维走一步，内存地址要跳多少
    - contiguity：数据是不是按某种连续顺序紧挨着放好了
  - 对 tensor 的常见操作只是改变 view 也就是底层 storage 的解释方式，而 clone()才会拷贝内存。
  - 因为物理内存和 view 的冲突，导致了每次取值的时候 都要加上步长，或者有些模型的做法干脆是直接拷贝一个contiguous的内存布局出来，再去取值。因为 contiguous() 是在必要时把这种逻辑顺序落实成新的物理排布。
  - contiguous这一步是耗费内存带宽的。


### autograd
- 自动计算梯度的过程，前向传播 forward 一遍，调用 `L.backward()` 后，从 L 出发，从后往前累积梯度，记录到 `.grad` 上。
- 从性能角度：
  - 前向传播阶段需要分配 grad 节点，并且多出了两遍计算（前向和后向）
  - 通过 `torch.no_grad()` 可以关掉 autograd 优化性能
### operator dispatch
- 类似 VFS，根据不同的 tensor 分发运算路径
- 每次 op 调用都要走这套路由：**Python 调用 → C++ dispatcher → 找到 kernel → launch**。对于计算量大的 op（matmul），这个 overhead 可以忽略。对于大量小 op 密集调用（elementwise、indexing），dispatch overhead 会积累成可测量的瓶颈。

## eager execution
- **Eager execution** 指的是：每一个 op 被调用时，**立刻执行**，立刻返回结果。没有延迟，没有先构建 graph 再执行的过程。基于前面的operator dispatch, 每一个op都要经过一系列的分发流程。
- 对于大量的小 op 这个过程会形成性能损耗。而PyTorch 默认就是 eager mode。也就是torch.xxx的操作默认以eager 的形式实时执行。逐行推进，没有"往前看"的能力。
- 为了改进，一般用 graph mode，也就是**先捕获，再编译，再执行。**
	- 这个编译可能融合了很多算子操作，内存搬运等，
	- 代价是第一次调用的耗时变长，以及compile 后不能直接debug
	- **原理：**
		- 捕获 FX Graph 的核心组件是 **TorchDynamo**，后面简称dynamo
		- 在Python 每个函数都会被编译成字节码后，Dynamo 用的是 CPython 提供的一个钩子 settrace, 在python的frame被调用时，先执行钩子，而不是执行字节码。以追踪函数调用的流程，使用假 tensor 模拟代码。记录下*"有一个 matmul，输入 shape 是这样，输出 shape 是那样"*
		- 这样就输出了计算图，但是他是静态的，只是记录这次的条件和if分支（如果下次if分支变了，就需要重新trace，也就是recompilation）
	-  FX Graph 构建完成后，就是交给后端 （默认是 Inductor 处理）做一些算子融合，内存规划（减少中间的buffer分配）
	- 然后 ```生成 Triton kernel（GPU）或 C++ kernel（CPU）```
	- 然后缓存编译结果，下次运行可以直接调用。
## 计算图可视化
有两种追踪层面
- fx 是给研究/编译器用的中间表示；
- jit.trace 解决的是另一个问题：把模型从 Python 环境里带走。
- fx
	- 不是真实数据，只是符号执行。
	- 这一步生成一个**计算图**，可以基于这个计算图修改其中的节点，叫做**图变换**，例如把RELU换成GELU。
	- 也可以将几个节点融合成一个cuda kernel，或者在计算图中插入量化节点
	- fx图是一个快照。
		- 静态的流程时（没有if等语句），生成图之后，你修改图、recompile、运行，就可以完全与代码解耦。
		- 有if的时候，如果修改节点可能会改变原有语义。
- jit.trace
	- 不再依赖python，用作训练结束后的部署
	- 模型结构，权重通过这种格式保存，在任意地方使用，c++也是通过这个方式加载的。
	- ```python
	    # 保存成单文件，包含模型结构 + 权重
	    torch.jit.save(traced, "mini_mlp.pt")
	  
	    # 任何地方加载，不需要 MiniMLP 类定义
	    loaded = torch.jit.load("mini_mlp.pt")
	    out = loaded(x)
	  ```
	
	  
