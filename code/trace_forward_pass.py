"""
Trace a minimal model's forward pass using two approaches:
  1. torch.fx.symbolic_trace  — static graph, shows the IR
  2. torch.jit.trace          — TorchScript, shows compiled ops
"""

import torch
import torch.nn as nn
import torch.fx as fx


# ── 1. Minimal model ────────────────────────────────────────────────────────
class MiniMLP(nn.Module):
    def __init__(self, in_dim: int = 8, hidden: int = 16, out_dim: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


model = MiniMLP()
x = torch.randn(2, 8)  # batch=2, in_dim=8


# ── 2. torch.fx symbolic trace ───────────────────────────────────────────────
print("=" * 60)
print("【torch.fx symbolic trace】")
traced_fx = fx.symbolic_trace(model)

# 打印每个节点: op / name / target / args
print(f"{'op':<12} {'name':<12} {'target':<20} args")
print("-" * 60)
for node in traced_fx.graph.nodes:
    print(f"{node.op:<12} {node.name:<12} {str(node.target):<20} {node.args}")

# 验证输出一致
out_fx = traced_fx(x)
print(f"\nOutput shape: {out_fx.shape}")


# ── 3. torch.jit.trace ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("【torch.jit.trace】")
traced_jit = torch.jit.trace(model, x)

# 打印 TorchScript IR graph
print(traced_jit.graph)

# 验证输出一致
out_jit = traced_jit(x)
print(f"Output shape: {out_jit.shape}")
assert torch.allclose(out_fx, out_jit, atol=1e-6), "outputs differ!"
print("Outputs match between fx and jit traces.")


# ── 4. 在 CUDA 上跑一次 (可选) ─────────────────────────────────────────────
if torch.cuda.is_available():
    print("\n" + "=" * 60)
    print("【CUDA forward pass】")
    model_gpu = model.cuda()
    x_gpu = x.cuda()
    out_gpu = model_gpu(x_gpu)
    print(f"Device: {out_gpu.device}, shape: {out_gpu.shape}")
