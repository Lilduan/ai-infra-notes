# Day 2

Date: 2026-03-18
Topic: PyTorch execution from a systems angle
Time spent: 60-90 min target

## Goal

从 **end-to-end critical path** 视角理解 PyTorch execution：区分哪些是 Python overhead，哪些是 backend execution，并开始把一次 forward pass 拆成 compute、memory movement、synchronization 三类成本。

## Today Checklist

- [ ] Review `tensor layout`、`autograd`、`operator dispatch`、`eager execution`
- [ ] Trace 一个最小 model 的 forward pass
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

- tensor layout
  - 类比 C 里的多维数组视图，有一些基于tensor的基本概念：
    - shape：每一维有多大
    - stride：沿着某一维走一步，内存地址要跳多少
    - contiguity：数据是不是按某种连续顺序紧挨着放好了
  - 对 tensor 的常见操作只是改变 view 也就是底层 storage 的解释方式，而 clone()才会拷贝内存。
  - 因为物理内存和 view 的冲突，导致了每次取值的时候 都要加上步长，或者有些模型的做法干脆是直接拷贝一个contiguous的内存布局出来，再去取值。因为 contiguous() 是在必要时把这种逻辑顺序落实成新的物理排布。
  - contiguous这一步是耗费内存带宽的。
- autograd
- operator dispatch
- eager execution





