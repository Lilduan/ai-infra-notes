# Day 2 - 从系统视角理解 PyTorch Execution

Date: 2026-03-18
Status: in progress

## Core Philosophy

PyTorch 不只是一个 tensor API。站在 AI infra 的视角，它本质上是一个 runtime stack：把 Python side 的 intent，转换成 backend side 的 execution。这个过程沿着一条明确的 critical path 展开，而这条路径主要由 `dispatch`、`memory movement`、`kernel launch`、`synchronization` 共同塑造。

真正该问的问题不是“我调用了哪个 API？”，而是 **“到底执行了什么，它在哪里执行，framework 在中间增加了哪些 overhead？”**

## 1. 今天要重新建立的主干概念

### Tensor Layout

- `contiguity`、`stride`、`shape metadata` 决定了一个 op 能不能直接读取数据，还是必须先做 `layout conversion`。
- 这件事很关键，因为 `layout mismatch` 往往会引入隐藏的 `memory movement`，而这正是很多 benchmark 初学者最容易漏掉的 infra cost。

### Autograd

- `autograd` 会在 `eager execution` 期间记录 computation graph，这样后续 `backward` 才能回放 gradient computation。
- 即使今天更偏向 inference-oriented execution，理解 `autograd` 仍然重要，因为它能帮助区分哪些工作属于 framework bookkeeping，哪些才是真正的 kernel work。

### Operator Dispatch

- 像 `torch.matmul(...)` 这样的 Python 调用，本身并不会直接执行数学计算。
- 它会先进入 `dispatcher`，解析 `dtype`、`device`、`layout`、`backend`，然后再选择具体实现。
- 也就是从这里开始，“PyTorch”不再只是 Python library，而开始表现为一个 runtime system。

### Eager Execution

- `eager mode` 会立刻执行 op，这对 debugging 非常友好，但也会比 compiled graph 更直接地暴露 `Python overhead`。
- 从 core philosophy 来看，`eager execution` 是一个明确的 tradeoff：它用 flexibility 和 transparency，交换了更少的 framework overhead amortization。

## 2. 一个最小 Forward Path

先用一个 tiny model，沿着下面这条路径去 trace：

```text
Python module call
  -> Python frame / argument handling
  -> PyTorch operator dispatch
  -> backend kernel selection
  -> tensor reads / writes
  -> result tensor materialization
  -> optional autograd bookkeeping
```

这条路径的价值在于：它迫使你把一次 forward pass 拆开看，而不是把所有时间都模糊地归为“模型在跑”。

## 3. Cost Breakdown Template

### Compute

- `matmul`、`convolution` 这类 dense math
- `elementwise arithmetic`

### Memory Movement

- 读取 weights 和 activations
- 写回 outputs
- `layout conversion` 或隐式 `copy`
- host side 创建 tensor，再把它送入 backend execution

### Synchronization

- 显式 `wait`
- 在 host 侧读取结果时触发的隐式 barrier
- 强制 execution ordering 的 profiling 或 debugging 操作

## 4. Python Overhead vs Backend Execution

### Python Overhead

- `module` / `function call overhead`
- Python 中的 `control flow`
- tensor orchestration 和 shape plumbing
- 在 kernel 之外完成的 `tokenization` 或 `preprocessing`

### Backend Execution

- 被 dispatch 的 `ATen ops`
- `BLAS` / backend kernels
- memory hierarchy 中真实发生的数据移动
- device side execution 以及相关的 runtime work

## 5. Open Questions

1. 在 `eager inference` 里，`operator dispatch` 具体从什么规模开始会成为可测量的 bottleneck？
2. 在常见的 Transformer path 里，哪些 `tensor layout mismatch` 最容易触发隐藏的 `copy`？
3. 当 gradients 被禁用后，`autograd-related bookkeeping` 还会保留多少？
4. 哪些 PyTorch profiler 视图最适合区分 `Python overhead` 和 backend kernel time？
5. 在 small-batch inference 下，最先主导总开销的到底是 `dispatch`、`memory movement`，还是 `kernel launch overhead`？

## 6. 跑完之后要补充的内容

- 一个具体的 forward example
- 一张简短的 execution path diagram
- 实际观察到的 hot ops
- 对可能发生过 `synchronization` 的位置做注释
