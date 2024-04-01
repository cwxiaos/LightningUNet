# 网络结构

```
Image ([B, 1, H, W])    // 原始图像
Image ([B, 3, H, W])    // 此算子预留给彩色图像, 如果是黑白图像则repeat为三通道图像, 此时三通道相同, 如果彩色图像可以利用三通道的信息

Image ([B, C, 512, 512])    // Zoom, 根据Network Config进行预处理, 如果原始图像已经是对应Shape, 则不需要预处理
// 预处理完成

PtachEmbedding ([B, H / 4 * W / 4, 96]) // 卷积到dim=96
                                        // 由于Lightning Attention的qkv dim =32(dim需要为2^n, dim>=32, dim>=64时OOM, 因此使用多头注意力,每头dim=32, heads=[3,6,12,24])

// 下采样

TransformerBlock(Lightning Attn()) x 2
PatchMerging()                      // ([B, H / 4 * W / 4, C]) -> ([B, H / 8 * W / 8, 2C])
TransformerBlock(Lightning Attn()) x 2
PatchMerging()                      // ([B, H / 8 * W / 8, 2C]) -> ([B, H / 16 * W / 16, 4C])
TransformerBlock(Lightning Attn()) x 2
PatchMerging()                      // ([B, H / 16 * W / 16, 4C]) -> ([B, H / 32 * W / 32, 8C])

TransformerBlock(Lightning Attn()) x2

// 上采样

PatchExpanding()                    // ([B, H / 32 * W / 32, 8C]) -> ([B, H / 16 * W / 16, 4C])
TransformerBlock(Lightning Attn())
PatchExpanding()                    // ([B, H / 16 * W / 16, 4C]) -> ([B, H / 8 * W / 8, 2C])
TransformerBlock(Lightning Attn())
PatchExpanding()                    // ([B, H / 8 * W / 8, 2C]) -> ([B, H / 4 * W / 4, C])
TransformerBlock(Lightning Attn())

PatchExpanding()                    // ([B, H / 4 * W / 4, C]) -> ([B, H / 4 * W / 4, 4C])

// Projection

Image ([B, 14, 512, 512])           // 14为数据集分类数
```

# PatchEmbedding

使用Swin Transformer的PatchEmbedding 

<!-- flatten: image([B, Channels, H, W]) &rarr; ([B, Channels, H * W]) -->
conv2d: ([B, Channels, H, W]) &rarr; ([B, Channels, H / P * W / P])

transpose: ([B, Channels, H / P * W/ P]) &rarr; ([B, H / P * W / P, Channels])

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

Forward时dim可以到较高的值, 在训练时如果qkv前面有linear层, dim只能取32, 64会导致OOM