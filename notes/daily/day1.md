  今天该完成的 4 件事：

  - [x] 写一页 note，回答 training infra / inference infra / platform
     infra 分别是什么。
  - [x] 列出性能维度：latency、throughput、memory、bandwidth、
     utilization、stability、cost，并给每个维度一句系统含义。
  - [x] 在你的 Mac 上把本地环境弄到可用：Python + PyTorch + profiler，
     至少能跑一个最小 tensor demo。
  - [x] 画出一次 inference request 的完整路径：input -> tokenize ->
     host preprocess -> model forward -> sampling -> output。



## infra的类别

- Training infra：
  - 在前期阶段，关注训练每一次weight的成本
  - 产物是模型权重，优化目标是训练每一次权重的开销
  - 优化 throughput / scale / distributed efficiency
- Inference infra：
  - 应用模型，通过预测或分类生成新输出结果的过程。
  - 产物是每一次回复用户请求的响应，优化目标是响应的性能和准确率。
  - 优化 latency / throughput / availability / cost
  - 一些性能指标：
    - 首 Token 延迟 (TTFT)：衡量系统生成第一个 token 所花费的时间
    - 吞吐量：从第二个 Token 开始的每个输出 Token 延迟(TPOT)，生成后续每个 token 的平均用时
    - 实际吞吐量：是在保持目标 TTFT 和 TPOT 前提下测量的吞吐量。
  - 不可能三角：
    - 延迟，减少预处理的batch size 或者 加大模型
    - 成本，加卡
    - 吞吐量，在控制成本的情况下，最大化AI的性能，意味着要思考很久，延迟高。

- 一次完整的推理流程：

  ![该图表展示了大语言模型中的模型推理流程](/Users/fine/Project/ai-infra-learning/notes/daily/assets/ai-inference-glossary-diagram.svg)



## 主要的性能指标

latency：请求从开始到结果的时间。

throughput，吞吐，可以按照请求，tokens，traning steps记

memory，举例：模型权重 14GB，KV cache 再吃 10GB，

bandwidth，数据搬运速率

utilization，实际工作的时间，而不是等待数据，阻塞

stability，比较宽泛

cost
