import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import Tensor

# helpers


def set_power_prop_global(power_prop_alpha):
    class PowerPropMixin:
        def __init__(self, *args, **kwargs):
            nonlocal power_prop_alpha
            super().__init__(*args, **kwargs)
            self.power_prop_alpha = power_prop_alpha

    class PowerLinear(PowerPropMixin, nn.Linear):
        def forward(self, input: Tensor) -> Tensor:
            params = self.weight * torch.pow(
                torch.abs(self.weight), self.power_prop_alpha - 1
            )
            return F.linear(input, params, self.bias)

    class PowerConv2d(PowerPropMixin, nn.Conv2d):
        def forward(self, input: Tensor) -> Tensor:
            params = self.weight * torch.pow(
                torch.abs(self.weight), self.power_prop_alpha - 1
            )
            return self._conv_forward(input, params, self.bias)

    nn.Linear = PowerLinear
    # nn.Conv2d = PowerConv2d


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class CNNEncoder(nn.Module):
    def __init__(
        self,
        kernel_size=7,
        stride=2,
        padding=0,
        pooling_kernel_size=7,
        pooling_stride=1,
        pooling_padding=1,
        activation=nn.ReLU(),
        n_filter_list=[1, 16, 32, 64],
        conv_bias=False,
    ):
        super(CNNEncoder, self).__init__()

        self.conv_block = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(
                        n_filter_list[i],
                        n_filter_list[i + 1],
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=conv_bias,
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=7, stride=1, padding=1),
                )
                for i in range(len(n_filter_list) - 1)
            ]
        )

    def forward(self, x):
        return self.conv_block(x)


class HybridViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=3,
        dim_head=64,
        dropout=0.0,
        n_filter_list=[1, 32, 64, 64],
        seq_pool=False,
        positional_embedding=True,
        emb_dropout=0.0,
    ):

        super().__init__()
        self.seq_pool = seq_pool
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 c) (p2 )",
                p1=patch_height,
                p2=patch_width,
            ),
        )

        self.encoder = CNNEncoder(n_filter_list=n_filter_list)

        self.to_flatten = nn.Sequential(
            Rearrange("b p c p1 p2-> b p (c p1 p2)"),
            nn.Linear(16 * 16 * n_filter_list[-1], dim),  # 4 or 16
        )

        if not seq_pool:
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        else:
            self.attention_pool = nn.Linear(dim, 1)

        if positional_embedding:
            self.positional_emb = nn.Parameter(
                torch.randn(1, num_patches + 1, dim), requires_grad=True
            )
            nn.init.trunc_normal_(self.positional_emb, std=0.2)
        else:
            self.positional_emb = None

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):

        x = self.to_patch_embedding(img)

        x = torch.cat(
            tuple(
                self.encoder(x[:, i : i + 1]).unsqueeze(dim=1)
                for i in range(x.shape[1])
            ),
            axis=1,
        )

        x = self.to_flatten(x)

        b, n, _ = x.shape
        if not self.seq_pool:
            cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.positional_emb is not None:
            x += self.positional_emb

        x = self.dropout(x)

        x = self.transformer(x)

        if self.seq_pool:
            x = torch.matmul(
                F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x
            ).squeeze(-2)
        else:
            x = x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
