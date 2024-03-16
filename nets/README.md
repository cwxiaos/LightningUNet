# 网络结构

由于Lightning Attn的限制, 网络只能做到三层, dim从32 -> 64 -> 128 -> 64 -> 32
如果需要更多层数, 在128之后增加Conv层处理到192

```
Image ([1, 1, H, W])    // 原始图像
Image ([1, 3, H, W])    // 如果是黑白图像则repeat为三通道图像

Image ([1, C, 512, 512])    // Zoom
// 预处理完成

PtachEmbedding ([1, H * W, 32]) // 卷积到32, 由于Lightning Attn的参数需要为2的幂且>=32
                                // 同时如果Dim过大(>=192), 会导致Out Of Memory(需要13G, torch限制10G)

// 下采样

TransformerBlock(Lightning Attn())  // ([1, H * W, 32]) -> ([1, H * W, 32])
PatchMerging()                      // ([1, H * W, 32]) -> ([1, H / 2 * W / 2, 64])
TransformerBlock(Lightning Attn())  // ([1, H / 2 * W / 2, 64]) -> ([1, H / 2 * W / 2, 64])
PatchMerging()                      // ([1, H / 2 * W / 2, 64]) -> ([1, H / 4 * W / 4, 128])

TransformerBlock(Lightning Attn())  // ([1, H / 4 * W / 4, 128]) -> ([1, H / 4 * W / 4, 128])

// 上采样

PatchExpanding()                    // ([1, H / 4 * W / 4, 128]) -> ([1, H / 2 * W / 2, 64])
TransformerBlock(Lightning Attn())  // ([1, H / 2 * W / 2, 64]) -> ([1, H / 2 * W / 2, 64])
PatchExpanding()                    // ([1, H / 2 * W / 2, 64]) -> ([1, H * W, 32])
TransformerBlock(Lightning Attn())  // ([1, H * W, 32]) -> ([1, H * W, 32])

// Projection

Image ([1, 14, 512, 512])           // 14为数据集分类数
```

# PatchEmbedding

使用Swin Transformer的PatchEmbedding 

flatten: image([1, Channels, H, W]) &rarr; ([1, Channels, H * W])

transpose: ([1, Channels, H * W]) &rarr; ([1, H * W, Channels])

[https://blog.csdn.net/lzzzzzzm/article/details/122902777](https://blog.csdn.net/lzzzzzzm/article/details/122902777)

# PatchExpanding

使用Swin Transformer的PatchExpanding

# Lightning Attention

[https://github.com/OpenNLPLab/lightning-attention](https://github.com/OpenNLPLab/lightning-attention)

```
import torch

from lightning_attn.ops import lightning_attn_func
from lightning_attn.utils import _build_slope_tensor

# dtype = torch.bfloat16
dtype = torch.float16
device = torch.device("cuda")
b, h, n, d, e = 2, 12, 2048, 192, 192

q = torch.randn((b, h, n, d), dtype=dtype, device=device).requires_grad_()
k = torch.randn((b, h, n, d), dtype=dtype, device=device).requires_grad_()
v = torch.randn((b, h, n, e), dtype=dtype, device=device).requires_grad_()
s = _build_slope_tensor(h).to(q.device).to(torch.float32)

o = lightning_attn_func(q, k, v, s)

print(o.shape)

loss = o.sum()
loss.backward()
```