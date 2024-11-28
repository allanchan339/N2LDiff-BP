from commons import *
import torch
from torch import nn
from utils.module_util import initialize_weights
import math
from flash_attn.modules.mha import FlashSelfAttention
import numbers

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
    
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FlashAttnImpl(nn.Module):
    def __init__(self, num_dims, num_heads, bias, flash_attn_valid_switch) -> None:
        super().__init__()
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.q1 = nn.Conv2d(num_dims, num_dims * 3, kernel_size=1, bias=bias)
        self.q2 = nn.Conv2d(num_dims * 3, num_dims * 3, kernel_size=3,
                            padding=1, groups=num_dims * 3, bias=bias)
        self.q3 = nn.Conv2d(num_dims * 3, num_dims * 3, kernel_size=3,
                            padding=1, groups=num_dims * 3, bias=bias)

        # self.fac = nn.Parameter(torch.ones(1))
        self.fin = nn.Conv2d(num_dims, num_dims, kernel_size=1, bias=bias)

        self.norm_3 = nn.InstanceNorm2d(num_dims * 3, affine=True)
        self.norm = nn.InstanceNorm2d(num_dims, affine=True)

        self.flash_attn = FlashSelfAttention(causal=False, softmax_scale=None, attention_dropout=0.0)

        self.flash_attn_valid_switch = flash_attn_valid_switch
        # self.attn = SelfAttention(causal=False, softmax_scale=None, attention_dropout=0.0)

    def forward(self, x):
        # x: [b, c, h, w]
        b, c, h, w = x.size()
        n_heads, dim_head = self.num_heads, c // self.num_heads
        qkv = self.norm_3(self.q3(self.q2(self.q1(x)))).view(b, -1, 3, n_heads, dim_head)


        if x.dtype == torch.float32:
            qkv = qkv.to(torch.bfloat16)
        
        if self.training or self.flash_attn_valid_switch:
            out = self.flash_attn(qkv)
        
        else:
            q, k, v = qkv.unbind(dim=2)
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])

            k.mul_(softmax_scale).transpose_(-2, -1)  # Scale in-place and then transpose

            # Matmul for bthd,bshd->bhts (equivalent to torch.einsum)
            out = torch.matmul(q, k)  # k is already transposed
            out = torch.softmax(out, dim=-1, dtype=v.dtype)
            out = torch.matmul(out, v)  # Matmul for bhts,bshd->bthd  
                  
        if x.dtype == torch.float32:
            out = out.to(torch.float32)

        out = self.fin(out.view(x.shape)) 
        out = self.norm(out)
        return out + x

class BPAttention(nn.Module):
    def __init__(self, num_dims, num_heads=1, bias=True, flash_attn_valid_switch=False, lambda_num = 1) -> None:
        super().__init__()
        assert num_dims % num_heads == 0
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.L1_att = FlashAttnImpl(num_dims, num_heads, bias, flash_attn_valid_switch)
        self.D1_att = FlashAttnImpl(num_dims, num_heads, bias, flash_attn_valid_switch)
        self.L2_att = FlashAttnImpl(num_dims, num_heads, bias, flash_attn_valid_switch)
        self.gelu = nn.GELU(approximate='tanh') 
        self.lambda_num = lambda_num
    def forward(self, x: torch.Tensor):
        assert len(x.size()) == 4
        
        y = self.L1_att(x)
        y = self.gelu(y)

        d = self.D1_att(y)
        d = self.gelu(d)

        r = self.L2_att(x-d)
        r = self.gelu(r)

        return x + y + r * self.lambda_num

class BPDualGateFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias=True):
        super().__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        hidden_features_twice = hidden_features*2

        self.project_in = nn.Conv2d(
            dim, hidden_features_twice, kernel_size=1, bias=bias)

        self.project_D = nn.Conv2d(hidden_features_twice, hidden_features_twice, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features_twice, bias=bias)
        
        self.project_L2 = nn.Conv2d(hidden_features_twice, hidden_features_twice, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features_twice, bias=bias)
        
        self.dwconv = nn.Conv2d(hidden_features_twice, hidden_features_twice, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features_twice, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)
        
        self.gelu = nn.GELU(approximate='tanh')

    def forward(self, x):
        xL1 = self.project_in(x)

        xD = self.project_D(xL1)

        xR = self.project_L2(xL1 - xD)

        x1, x2 = self.dwconv(xL1 + xR).chunk(2, dim=1)

        return self.project_out(self.gelu(x2)*x1 + self.gelu(x1)*x2) + x

class BPTransformerBlock(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads=1, 
                 ffn_expansion_factor=2.66, 
                 bias=True, 
                 LayerNorm_type='WithBias', 
                 time_emb_dim=0, 
                 flash_attn_valid_switch=False, 
                lambda_num = 1
                ):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        if self.time_emb_dim > 0:
            self.mlp = nn.Sequential(
                nn.GELU(approximate='tanh'),
                nn.Linear(time_emb_dim, dim)
            )
        else:
            self.mlp = nn.Identity()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = BPAttention(dim, num_heads, bias, flash_attn_valid_switch, lambda_num)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = BPDualGateFeedForward(dim, ffn_expansion_factor, False)

    def forward(self, x, t):
        x = x + self.attn(self.norm1(x))

        if t is not None and self.time_emb_dim > 0:
            x += self.mlp(t)[:, :, None, None]

        x = x + self.ffn(self.norm2(x))
        return x, t

#### BP Attention Fusion Block
class CrossAttn(nn.Module):  
    """ Layer attention module"""
    def __init__(self, in_dim,bias=True):
        super(CrossAttn, self).__init__()
        self.chanel_in = in_dim

        self.temperature = nn.Parameter(torch.ones(1))

        self.qkv = nn.Conv2d( self.chanel_in ,  self.chanel_in *3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(self.chanel_in*3, self.chanel_in*3, kernel_size=3, stride=1, padding=1, groups=self.chanel_in*3, bias=bias)
        self.project_out = nn.Conv2d(self.chanel_in, self.chanel_in, kernel_size=1, bias=bias)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()

        x_input = x.view(m_batchsize,N*C, height, width)
        qkv = self.qkv_dwconv(self.qkv(x_input))
        q, k, v = qkv.chunk(3, dim=1)

        q = torch.nn.functional.normalize(q.view(m_batchsize, N, -1), dim=-1)
        k = torch.nn.functional.normalize(k.view(m_batchsize, N, -1), dim=-1)
        v = v.view(m_batchsize, N, -1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = self.project_out((attn @ v).view(m_batchsize, -1, height, width))
        out = (out.view(m_batchsize, N, C, height, width) + x).view(m_batchsize, -1, height, width)

        return out

##########################################
##---------- Squential -------------------

##########################################################################
## Embedding modules
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj1 = nn.Conv2d(in_c, (in_c+embed_dim)//2, kernel_size=3, stride=1, padding=1, bias=bias)
        self.proj2 = nn.Conv2d((in_c+embed_dim)//2, (in_c+embed_dim)//2, kernel_size=3, stride=1, padding=1, bias=bias)
        self.proj3 = nn.Conv2d((in_c+embed_dim)//2, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
        
    def forward(self, x):
        x = self.proj1(x)
        x = self.proj2(x)
        x = self.proj3(x)

        return x

class CustomSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class ViT(nn.Module):
    def __init__(self,
        inp_channels=128,
        out_channels=128,
        dim = 128//2,
        time_emb_dim = 0,
        num_blocks = [1,3,6,12],
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = True,
        LayerNorm_type = 'WithBias',
        skip = False,
        flash_attn_valid_switch = False,
        lambda_num = 1
) -> None:
        super().__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_1 = CustomSequential(*[BPTransformerBlock(
            dim=dim, 
            num_heads=heads[0], 
            ffn_expansion_factor=ffn_expansion_factor, 
            bias=bias, 
            LayerNorm_type=LayerNorm_type, 
            time_emb_dim=time_emb_dim,
            flash_attn_valid_switch=flash_attn_valid_switch,
            lambda_num = lambda_num
            ) for i in range(num_blocks[0])]
            )

        self.encoder_2 = CustomSequential(*[BPTransformerBlock(
            dim=int(dim), 
            num_heads=heads[1], 
            ffn_expansion_factor=ffn_expansion_factor, 
            bias=bias, 
            LayerNorm_type=LayerNorm_type, 
            time_emb_dim=time_emb_dim,
            flash_attn_valid_switch=flash_attn_valid_switch,
            lambda_num = lambda_num
            ) for i in range(num_blocks[1])])

        self.encoder_3 = CustomSequential(*[BPTransformerBlock(
            dim=int(dim), 
            num_heads=heads[2], 
            ffn_expansion_factor=ffn_expansion_factor, 
            bias=bias, 
            LayerNorm_type=LayerNorm_type, 
            time_emb_dim=time_emb_dim,
            flash_attn_valid_switch=flash_attn_valid_switch,
            lambda_num = lambda_num
            ) for i in range(num_blocks[2])])

        self.layer_fussion = CrossAttn(in_dim=int(dim*3))
        self.conv_fuss = nn.Conv2d(int(dim * 3), int(dim), kernel_size=1, bias=False)

        self.output = OverlapPatchEmbed(in_c=int(dim), embed_dim=out_channels, bias=False)

        self.skip = skip

    def forward(self, x, t):
        # x: [b, 128, h, w]
        inp_enc_encoder1 = self.patch_embed(x) 

        out_enc_encoder1, _ = self.encoder_1(inp_enc_encoder1, t)
        out_enc_encoder2, _ = self.encoder_2(out_enc_encoder1, t)
        out_enc_encoder3, _ = self.encoder_3(out_enc_encoder2, t)

        inp_fusion_123 = torch.cat([out_enc_encoder1.unsqueeze(1),out_enc_encoder2.unsqueeze(1),out_enc_encoder3.unsqueeze(1)],dim=1)

        out_fusion_123 = self.layer_fussion(inp_fusion_123)
        out_fusion_123 = self.conv_fuss(out_fusion_123)

        if self.skip:
            out = self.output(out_fusion_123)+ x
        else:
            out = self.output(out_fusion_123)

        return out


class UViT(nn.Module):
    def __init__(self, 
                 unet_dim, 
                 unet_outdim, 
                 dim_mults, 
                 use_ViT, 
                 use_wn, 
                 use_instance_norm, 
                 weight_init, 
                 stronger_cond, 
                 in_dim = 3, 
                 dim_adjust_factor = None, 
                 num_blocks=[1,3,6,12], 
                 heads=[1,2,4,8], 
                 ffn_expansion_factor=2.66, 
                 bias=True, 
                 LayerNorm_type='WithBias', 
                 skip=False,
                 flash_attn_valid_switch=False,
                lambda_num = 1

                 ):
        super().__init__()

        dim = unet_dim
        out_dim = unet_outdim
        dim_mults = dim_mults
        in_dim = in_dim
        dims = [in_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        groups = 0
        self.use_ViT = use_ViT
        self.use_wn = use_wn
        self.use_in = use_instance_norm
        self.weight_init = weight_init
        self.stronger_cond = stronger_cond
        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        if self.stronger_cond:
            dim_adjust = 2
        else:
            dim_adjust = 1
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=dim, groups=groups, use_in = self.use_in),
                ResnetBlock(dim_out * dim_adjust, dim_out, time_emb_dim=dim, groups=groups, use_in = self.use_in),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))
             
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(
            mid_dim, mid_dim, time_emb_dim=dim, groups=groups, use_in = self.use_in)
        
        if self.use_ViT:
            if dim_adjust_factor is None:
                dim_adjust_factor = 192 * heads[0] / mid_dim # flash attn only allow <=192 for 4090, <=256 for A100

            self.ViT = ViT(inp_channels=mid_dim,
                           out_channels=mid_dim,
                           dim = int(mid_dim * dim_adjust_factor),
                           time_emb_dim=0,
                           num_blocks = num_blocks,
                           heads=heads,
                            ffn_expansion_factor=ffn_expansion_factor,
                            bias=bias,
                            LayerNorm_type=LayerNorm_type,
                            skip=skip,
                            flash_attn_valid_switch=flash_attn_valid_switch,
                            lambda_num = lambda_num
                            )
        else: 
            self.ViT = nn.Identity()

        self.mid_block2 = ResnetBlock(
            mid_dim, mid_dim, time_emb_dim=dim, groups=groups, use_in = self.use_in)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in,
                            time_emb_dim=dim, groups=groups, use_in = self.use_in),
                ResnetBlock(dim_in, dim_in, time_emb_dim=dim, groups=groups, use_in = self.use_in),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            Block(dim, dim, groups=groups),
            nn.Conv2d(dim, out_dim, 1)
        )

        # if hparams['res'] and hparams['up_input']:
        # self.up_proj = nn.Sequential(
        #         nn.ReflectionPad2d(1), nn.Conv2d(3, dim, 3),
        #     )
        if self.use_wn:
            self.apply_weight_norm()
        if self.weight_init:
            self.apply(initialize_weights)

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                # print(f"| Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)
    
    def apply_layer_norm(self):
        def _apply_layer_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.layer_norm(m)
                print(f"| Weight norm is applied to {m}.")
    
    def forward(self, x, time, cond, feats_cond):
        t = self.time_pos_emb(time)
        t = self.mlp(t)

        h = []
        x = torch.cat((x, cond), dim=1)
        # cond = torch.cat(cond[2::4], 1) # from rrdb net
        # cond = self.cond_proj(torch.cat(cond[2::4], 1)) # cond[start at 2 -> every third item we take], finally get [20, 32*3, 20, 20]
        for i, (resnet, resnet2, downsample) in enumerate(self.downs):
            x = resnet(x, t)
            if self.stronger_cond:
                x = torch.cat((x, feats_cond[i]), dim=1)
            x = resnet2(x, t)
            # if i == 0:
                # x = x + cond
                # if hparams['res'] and hparams['up_input']:
                # x = x + self.up_proj(img_lr_up)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t) #output [2,128,20,20]
        x = self.ViT(x, t)
        x = self.mid_block2(x, t)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.final_conv(x)

        return x, t

    def make_generation_fast_(self):
        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(remove_weight_norm)

if __name__ == '__main__':
    input_size = 160
    
    model = UViT(
        unet_dim = 64, 
        unet_outdim = 3, 
        dim_mults = [1, 2, 2, 4, 4], 
        use_ViT = True, 
        use_wn = True, 
        use_instance_norm = True, 
        weight_init = True, 
        stronger_cond = False, 
        in_dim = 6,
        dim_adjust_factor=1.5,
        num_blocks=[1,3,6,12],
        heads=[2,4,8,16],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        skip=False
    )

    model.cuda()
    x = torch.randn(2, 3, 608, 400).to('cuda')
    cond = torch.randn(2, 3, 608, 400).to('cuda')

    t = torch.randint(0, 100, (x.shape[0],)).to('cuda')
    out, _ = model(x, t, cond, feats_cond=None) # same shape with x
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss = torch.mean(out - torch.randn_like(out))
    opt.zero_grad() # clear previous gradients
    loss.backward() # compute gradients of all variables wrt loss
    opt.step()
    print(out.shape)