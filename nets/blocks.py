import torch
import torch.nn as nn

from lightning_attn.ops import lightning_attn_func
from lightning_attn.utils import _build_slope_tensor

from timm.models.layers import DropPath


class LightningAttention(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads=1,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * input_resolution[0] - 1) * (2 * input_resolution[1] - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        # print(f" ___ shape :{self.relative_position_bias_table.shape}")
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, l, c = x.shape
        # nn.functional.pad(x, (0, 32 - c), 'constant', 0)s
        qkv = self.qkv(x).reshape(b, l, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        print(qkv.shape)
        q, k, v = qkv[0], qkv[1], qkv[2]

        print(f"[I] {q.shape}, {k.shape}, {v.shape}")

        q = q * self.scale
        attn = lightning_attn_func(q, k, v, self.relative_position_bias_table)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, l, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


# class PatchExpand(nn.Module):
#     def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.input_resolution = input_resolution
#         self.dim = dim
#         self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
#         self.norm = norm_layer(dim // dim_scale)
#
#     def forward(self, x):
#         """
#         x: B, H*W, C
#         """
#         H, W = self.input_resolution
#         x = self.expand(x)
#         B, L, C = x.shape
#         assert L == H * W, "input feature has wrong size"
#
#         x = x.view(B, H, W, C)
#         x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
#         x = x.view(B,-1,C//4)
#         x= self.norm(x)
#
#         return x

class TransformerBlock(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.attn = LightningAttention(
            dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x):
        h, w = self.input_resolution
        b, l, c = x.shape
        assert l == h * w, f"Input feature has wrong size: {l} != ({h}*{w})"

        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_path(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + shortcut

        return x


class LayerDownSample(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 qkv_bias,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 down_sample=None):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                input_resolution=[input_resolution[0] // (2 ** (i + 2)), input_resolution[1] // (2 ** (i + 2))],
                num_heads=num_heads,
                mlp_ratio=4,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            ) for i in range(depth)
        ])

        if down_sample is not None:
            self.down_sample = down_sample(input_resolution, dim)
        else:
            self.down_sample = None

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        if self.down_sample is not None:
            x = self.down_sample(x)
        return x


class LayerUpSample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


# class PatchEmbedding1(nn.Module):
#     r"""
#     Patch Embedding
#     Lightning Attn Can Handle Long Sequences, So Input is Flattened Directly
#     # TODO: Add Patch Embedding based on Patch Size

#     Args:
#         image_size: int, Input image size
#         patch_size: int, Patch size
#         in_channels: int, Number of input channels
#         embed_dim: int, Embedding dimension
#         dropout: float, Dropout rate
#     """

#     def __init__(self, image_size, patch_size, in_channels, embed_dim, dropout):
#         super().__init__()
#         self.image_size = image_size
#         self.patch_size = patch_size
#         self.in_channels = in_channels
#         self.embed_dim = embed_dim
#         self.dropout = dropout

#         patches_resolution = [image_size // patch_size, image_size // patch_size]
#         self.num_patches = patches_resolution[0] * patches_resolution[1]

#     def forward(self, x):
#         b, c, h, w = x.shape
#         assert h == self.image_size and w == self.image_size, f"Image Shape Doesn't Match: {h}x{w} != {self.image_size}x{self.image_size}"
#         x = x.flatten(2)  # B, C, H*W
#         x = x.transpose(1, 2)  # B, H*W, C
#         return x


class PatchEmbedding(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = [img_size, img_size]
        patch_size = [patch_size, patch_size]
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    # ([1, 3, 224, 224]) -> ([1, 3136, 96])
    def forward(self, x):
        B, C, H, W = x.shape
        # TODO look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

class UNet(nn.Module):
    r"""
    UNet

    Args:
        image_size: int, Input image size
        patch_size: int, Patch size
        in_channels: int, Number of input channels
        embed_dim: int, Embedding dimension
        dropout: float, Dropout rate
        ape: bool, Absolute position embedding
    """

    def __init__(self,
                 embed_dim,
                 image_size=512,
                 patch_size=1,
                 in_channels=3,
                 depths=(2, 2, 2, 2),
                 dropout=0.,
                 ape=False):

        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_layers = len(depths)
        self.dropout = dropout
        self.ape = ape

        self.patch_embedding = PatchEmbedding(img_size=image_size,
                                                patch_size=patch_size,
                                                in_chans=in_channels,
                                                embed_dim=embed_dim)
        num_patches = self.patch_embedding.num_patches

        # TODO: Implement Absolute Position Embedding
        if self.ape:
            # self.absolute_position_embedding = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            self.absolute_position_embedding = nn.Parameter(torch.zeros(1, num_patches, 3))
            nn.init.trunc_normal_(self.absolute_position_embedding, std=0.02)

        self.pos_drop = nn.Dropout(p=dropout)

        # stochastic depth
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers_downSample = nn.ModuleList()
        for i in range(self.num_layers):
            layer = LayerDownSample(
                dim=int(embed_dim * 2 ** i),
                input_resolution=(image_size // (2 ** i), image_size // (2 ** i)),
                depth=depths[i],
                num_heads=1,
                qkv_bias=True,
                qk_scale=None,
                drop=0.,
                attn_drop=0.,
                drop_path=0.,
                norm_layer=nn.LayerNorm,
                down_sample=PatchMerging if (i < self.num_layers - 1) else None
            )
            self.layers_downSample.append(layer)

        self.layers_upSample = nn.ModuleList()
        for i in range(self.num_layers):
            layer = LayerUpSample()
            self.layers_upSample.append(layer)

    def forward(self, x):
        print(f"___ x.shape: {x.shape}")
        x = self.patch_embedding(x)
        print(f"___ x.shape: {x.shape}")

        if self.ape:
            x += self.absolute_position_embedding
        x = self.pos_drop(x)

        x_down_sample = []
        for layer in self.layers_downSample:
            x_down_sample.append(x)
            x = layer(x)

        # for i, layer in enumerate(self.layers_upSample):
        #     if i:
        #         x = torch.concat([x, x_down_sample.pop()], dim=1)
        #     x = layer(x)

        return x
