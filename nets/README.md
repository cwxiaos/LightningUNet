# PatchEmbedding

由于Lightning Attn可以处理超长序列, 对输入图像直接做flatten, 用Transformer直接感知全局Context

其中patch_size参数为预留参数, 方便后期修改PatchEmbedding的实现方式

flatten: image([1, Channels, H, W]) &rarr; ([1, Channels, H * W])

transpose: ([1, Channels, H * W]) &rarr; ([1, H * W, Channels])

[https://blog.csdn.net/lzzzzzzm/article/details/122902777](https://blog.csdn.net/lzzzzzzm/article/details/122902777)


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