import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from src.backbones import InceptionV3
from src.attention import SelfAttention, Linear_global
# helpers

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
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Image2Patch_Embedding(nn.Module):
    def __init__(self, patch_size, channels, dim):
        super().__init__()
        patch_height, patch_width = pair(patch_size)
        patch_dim = channels * patch_height * patch_width

        self.im2patch = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width)
        self.patch2latentv = nn.Linear(patch_dim, dim)

    def forward(self, x):
        x = self.im2patch(x)
        x = self.patch2latentv(x)
        return x

class Latentv2Image(nn.Module):
    def __init__(self, patch_size, channels, dim):
        patch_height, patch_width = pair(patch_size)
        self.latentv2patch = nn.Linear(dim, channels*patch_height*patch_width)
        self.vec2square = Rearrange('(c h w) -> c h w', c = channels, h = patch_height, w = patch_width)

        
    def forward(self, x):
        return x


class ReconstructionConv(nn.Module):
    def __init__(self, dim, hidden_channels, patch_size, image_size, cnn_depth, channels=3):
        super().__init__()
        self.dim = dim
        self.image_size = image_size
        self.patch_size = patch_size

        patch_height, patch_width = pair(patch_size)
        self.patch_dim = channels * patch_height * patch_width 

        self.latent2patchv = nn.Linear(in_features=self.dim, out_features=self.patch_dim)
        self.patchv2patch = Rearrange('b p (c h w) -> b p c h w', c=channels, h=patch_height, w=patch_width)
        self.net = InceptionV3(None)
        self.embedding = Image2Patch_Embedding(patch_size=patch_height, channels=channels, dim=dim)
        self.fc = nn.Linear(in_features=self.dim, out_features=self.dim)

    def reconstruct(self, patchs):
        image_height, image_width = pair(self.image_size)
        patch_height, patch_width = pair(self.patch_size)
        h = int(image_height / patch_height)
        w = int(image_width / patch_width)

        images = []
        for i in range(h):
            raw = []
            for j in range(w):
                raw.append(patchs[:, i*h + j, :, :])
            raw = torch.cat(raw, dim=3)
            images.append(raw)
        images = torch.cat(images, dim=2)
        return images

    def forward(self, x):
        x = self.latent2patchv(x)
        x = self.patchv2patch(x)
        x = self.reconstruct(x)

        x = self.net(x)

        return F.normalize(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class ConvolutionalTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, hidden_channels, patch_size, image_size, cnn_depth, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim, dropout=dropout))
            ]))
        
        self.reconstruct = ReconstructionConv(dim=dim, hidden_channels=hidden_channels, patch_size=patch_size, image_size=image_size, cnn_depth=cnn_depth)
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            identity = x
            x = ff(x) + identity
        x = self.reconstruct(x)
        return x

class VisionConformer(nn.Module):
    def __init__(
            self,
            *, 
            image_size=(224, 224),
            patch_size=(16, 16),
            dim = 128,
            depth = 2,
            heads = 4,
            mlp_dim = 256,
            channels = 3,
            dim_head = 64,
            dropout = 0.1,
            emb_dropout = 0.1,
            hidden_channels,
            cnn_depth):

        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.to_patch_embedding = Image2Patch_Embedding(patch_size=patch_size, channels=channels, dim=dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = ConvolutionalTransformer(dim=dim, depth=depth, heads=heads, dim_head=dim_head,
                                       mlp_dim=mlp_dim, dropout=dropout, hidden_channels=hidden_channels,
                                       patch_size=patch_size, image_size=image_size, cnn_depth=cnn_depth)


        self.attn = SelfAttention(None)
        self.linear = Linear_global(64)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.attn(x)
        x = self.linear(x)
        return x

if __name__ == "__main__":
    image_size = (224, 224)       # Kích thước ảnh (cao, rộng)
    patch_size = (16, 16)       # Kích thước mỗi patch
    num_classes = 10            # Số lớp đầu ra
    dim = 128                   # Chiều không gian ẩn
    depth = 4                   # Số tầng Transformer
    heads = 4                   # Số đầu attention
    mlp_dim = 256               # Hidden size trong FeedForward
    dim_head = 32               # Kích thước mỗi head trong attention
    hidden_channels = 64        # Số kênh ẩn trong CNN
    cnn_depth = 1               # Số tầng CNN trong ReconstructionConv
    batch_size = 2              # Batch size

    # Tạo ảnh giả lập (batch_size, 3, H, W)
    dummy_input = torch.randn(batch_size, 3, *image_size)

    # Khởi tạo mô hình
    model = VisionConformer(
        image_size=image_size,
        patch_size=patch_size,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dim_head=dim_head,
        hidden_channels=hidden_channels,
        cnn_depth=cnn_depth
    )

    # Chạy forward
    output = model(dummy_input)

    # In kết quả
    print("Output shape:", output.shape)