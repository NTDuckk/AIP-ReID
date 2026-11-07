import torch
import torch.nn as nn
import numpy as np
from .clip.model import Transformer, LayerNorm
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0], device=x.device), tokenized_prompts.argmax(dim=-1).to(x.device)] @ self.text_projection
        return x

class IM2TEXT(nn.Module):
    def __init__(self, embed_dim=512, middle_dim=512, output_dim=512, n_layer=3, dropout=0.1):
        super().__init__()
        self.fc_out = nn.Linear(middle_dim, output_dim)
        layers = []
        dim = embed_dim
        for _ in range(n_layer):
            block = []
            block.append(nn.Linear(dim, middle_dim))
            block.append(nn.Dropout(dropout))
            block.append(nn.ReLU())
            dim = middle_dim
            layers.append(nn.Sequential(*block))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)


def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model

from .clip import clip

def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model

class InversionPromptLearner3(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.prompt_clothes = IM2TEXT(
            embed_dim=clip_model.visual.output_dim,
            hidden_dim=1024,
            output_dim=clip_model.transformer.width
        )
        self.prompt_hairstyle = IM2TEXT(
            embed_dim=clip_model.visual.output_dim,
            hidden_dim=1024,
            output_dim=clip_model.transformer.width
        )
        self.prompt_carrying = IM2TEXT(
            embed_dim=clip_model.visual.output_dim,
            hidden_dim=1024,
            output_dim=clip_model.transformer.width
        )

    def forward(self, image_feature):
        clothes_token = self.prompt_clothes(image_feature).unsqueeze(1)
        hairstyle_token = self.prompt_hairstyle(image_feature).unsqueeze(1)
        carrying_token = self.prompt_carrying(image_feature).unsqueeze(1)
        return clothes_token, hairstyle_token, carrying_token

class InversionPromptLearner5(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.prompt_top = IM2TEXT(
            embed_dim=clip_model.visual.output_dim,
            middle_dim=512,
            output_dim=clip_model.transformer.width,
            n_layer=3,
            dropout=0.1
        )
        self.prompt_underneath = IM2TEXT(
            embed_dim=clip_model.visual.output_dim,
            middle_dim=512,
            output_dim=clip_model.transformer.width,
            n_layer=3,
            dropout=0.1
        )
        self.prompt_shoes = IM2TEXT(
            embed_dim=clip_model.visual.output_dim,
            middle_dim=512,
            output_dim=clip_model.transformer.width,
            n_layer=3,
            dropout=0.1
        )
        self.prompt_hairstyle = IM2TEXT(
            embed_dim=clip_model.visual.output_dim,
            middle_dim=512,
            output_dim=clip_model.transformer.width,
            n_layer=3,
            dropout=0.1
        )
        self.prompt_carrying = IM2TEXT(
            embed_dim=clip_model.visual.output_dim,
            middle_dim=512,
            output_dim=clip_model.transformer.width,
            n_layer=3,
            dropout=0.1
        )
        
    def forward(self, image_feature):
        top_token = self.prompt_top(image_feature).unsqueeze(1)
        underneath_token = self.prompt_underneath(image_feature).unsqueeze(1)
        shoes_token = self.prompt_shoes(image_feature).unsqueeze(1)
        hairstyle_token = self.prompt_hairstyle(image_feature).unsqueeze(1)
        carrying_token = self.prompt_carrying(image_feature).unsqueeze(1)
        return top_token, underneath_token, shoes_token, hairstyle_token, carrying_token

class Prompt_Cat3(nn.Module):
    """
    Prompt learner for 3 attributes: clothes, hairstyle, carrying.
    Uses a fixed sentence with 3 X placeholders, and inserts attribute tokens at those positions.
    """
    def __init__(self, dataset_name, dtype, token_embedding):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            ctx_init = "A photo of a vehicle wearing X clothes, having X hairstyle and carrying X."
        else:
            ctx_init = "A photo of a person wearing X clothes, having X hairstyle and carrying X."

        ctx_dim = 512
        ctx_init = ctx_init.replace("_", " ")
        tokenized_prompts = clip.tokenize(ctx_init).to(token_embedding.weight.device)
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts

        # Find positions of X tokens (token id 343 in CLIP)
        x_token_id = 343
        token_ids = tokenized_prompts[0].tolist()
        x_positions = [i for i, t in enumerate(token_ids) if t == x_token_id]
        if len(x_positions) != 3:
            raise RuntimeError(f"Expected 3 X tokens, found {len(x_positions)} at {x_positions}")

        # Register buffer segments
        self.register_buffer("token_prefix_0", embedding[:, :x_positions[0], :])        # before first X
        self.register_buffer("token_prefix_1", embedding[:, x_positions[0]+1:x_positions[1], :])  # between X1 and X2
        self.register_buffer("token_prefix_2", embedding[:, x_positions[1]+1:x_positions[2], :])  # between X2 and X3
        self.register_buffer("token_suffix", embedding[:, x_positions[2]+1:, :])                 # after last X

    def forward(self, label, prom_list):
        # prom_list: list of 3 tensors, each shape [B, 1, D] or [B, D]
        if prom_list is None or len(prom_list) != 3:
            raise ValueError("prom_list must be a list/tuple of length 3")

        b = label.shape[0]
        formatted = []
        for i, token in enumerate(prom_list):
            if token.dim() == 2:
                token = token.unsqueeze(1)
            elif token.dim() != 3:
                raise ValueError(f"prom_list[{i}] must be 2D or 3D, got {token.dim()}D")
            if token.size(2) != self.token_prefix_0.size(2):
                raise ValueError(f"prom_list[{i}] last dim must be {self.token_prefix_0.size(2)}")
            formatted.append(token.to(device=self.token_prefix_0.device, dtype=self.token_prefix_0.dtype))

        # Build prompt sequence
        pieces = [self.token_prefix_0.expand(b, -1, -1)]
        for i, token in enumerate(formatted):
            pieces.append(token)
            if i < len(formatted) - 1:
                pieces.append(getattr(self, f"token_prefix_{i+1}").expand(b, -1, -1))
        pieces.append(self.token_suffix.expand(b, -1, -1))

        prompts = torch.cat(pieces, dim=1)
        return prompts

# class Prompt_Cat5(nn.Module):
#     """
#     Prompt learner for 5 attributes: top, underneath, shoes, hairstyle, carrying.
#     Uses a fixed sentence with 5 X placeholders, and inserts attribute tokens at those positions.
#     """
#     def __init__(self, dataset_name, dtype, token_embedding):
#         super().__init__()
#         if dataset_name == "VehicleID" or dataset_name == "veri":
#             ctx_init = "A photo of a vehicle wearing X on top, X underneath, X shoes, having X hairstyle and carrying X."
#         else:
#             ctx_init = "A photo of a person wearing X on top, X underneath, X shoes, having X hairstyle and carrying X."

#         ctx_dim = 512
#         tokenized_prompts = clip.tokenize(ctx_init).cuda()
#         with torch.no_grad():
#             embedding = token_embedding(tokenized_prompts).type(dtype)
#         self.tokenized_prompts = tokenized_prompts

#         # Find positions of X tokens
#         x_token_id = 343
#         token_ids = tokenized_prompts[0].tolist()
#         x_positions = [i for i, t in enumerate(token_ids) if t == x_token_id]
#         if len(x_positions) != 5:
#             raise RuntimeError(f"Expected 5 X tokens, found {len(x_positions)} at {x_positions}")

#         # Register buffer segments
#         self.register_buffer("token_prefix_0", embedding[:, :x_positions[0], :])        # before first X
#         self.register_buffer("token_prefix_1", embedding[:, x_positions[0]+1:x_positions[1], :])  # between X1 and X2
#         self.register_buffer("token_prefix_2", embedding[:, x_positions[1]+1:x_positions[2], :])  # between X2 and X3
#         self.register_buffer("token_prefix_3", embedding[:, x_positions[2]+1:x_positions[3], :])  # between X3 and X4
#         self.register_buffer("token_prefix_4", embedding[:, x_positions[3]+1:x_positions[4], :])  # between X4 and X5
#         self.register_buffer("token_suffix", embedding[:, x_positions[4]+1:, :])                 # after last X
#     def forward(self, label, prom_list):
#         # prom_list: list of 5 tensors, each shape [B, 1, D] or [B, D]
#         if prom_list is None or len(prom_list) != 5:
#             raise ValueError("prom_list must be a list/tuple of length 5")

#         b = label.shape[0]
#         formatted = []
#         for i, token in enumerate(prom_list):
#             if token.dim() == 2:
#                 token = token.unsqueeze(1)
#             elif token.dim() != 3:
#                 raise ValueError(f"prom_list[{i}] must be 2D or 3D, got {token.dim()}D")
#             if token.size(2) != self.token_prefix_0.size(2):
#                 raise ValueError(f"prom_list[{i}] last dim must be {self.token_prefix_0.size(2)}")
#             formatted.append(token.to(device=self.token_prefix_0.device, dtype=self.token_prefix_0.dtype))

#         # Build prompt sequence
#         pieces = [self.token_prefix_0.expand(b, -1, -1)]
#         for i, token in enumerate(formatted):
#             pieces.append(token)
#             if i < len(formatted) - 1:
#                 pieces.append(getattr(self, f"token_prefix_{i+1}").expand(b, -1, -1))
#         pieces.append(self.token_suffix.expand(b, -1, -1))

#         prompts = torch.cat(pieces, dim=1)
#         return prompts

#------ debug---------
# class Prompt_Cat5(nn.Module):
#     """
#     Prompt template with 5 X placeholders.
#     Insert 5 attribute tokens directly into the X positions.
#     """
#     def __init__(self, dataset_name, dtype, token_embedding):
#         super().__init__()
#         if dataset_name == "VehicleID" or dataset_name == "veri":
#             template = "A photo of a vehicle wearing X on top, X underneath, X shoes, having X hairstyle and carrying X."
#         else:
#             template = "A photo of a person wearing X on top, X underneath, X shoes, having X hairstyle and carrying X."

#         self.num_attributes = 5

#         tokenized_prompts = clip.tokenize(template).cuda()
#         with torch.no_grad():
#             embedding = token_embedding(tokenized_prompts).type(dtype)

#         self.tokenized_prompts = tokenized_prompts
#         self.register_buffer("template_embedding", embedding)   # [1, 77, ctx_dim]

#         x_token_id = clip.tokenize("X")[0, 1].item()
#         x_positions = (tokenized_prompts[0] == x_token_id).nonzero(as_tuple=False).view(-1)

#         if x_positions.shape[0] != self.num_attributes:
#             raise RuntimeError(
#                 f"Expected {self.num_attributes} X positions, got {x_positions.shape[0]}"
#             )

#         self.register_buffer("x_positions", x_positions)
#         self.dtype = dtype

#     def forward(self, prom_list):
#         if prom_list is None or len(prom_list) != self.num_attributes:
#             raise ValueError(f"prom_list must be a list/tuple of length {self.num_attributes}")

#         first = prom_list[0]
#         if first.dim() == 3:
#             B = first.shape[0]
#         elif first.dim() == 2:
#             B = first.shape[0]
#         else:
#             raise ValueError(f"prom_list[0] must be 2D or 3D, got {first.dim()}D")

#         prompts = self.template_embedding.expand(B, -1, -1).clone()

#         for i, token in enumerate(prom_list):
#             if token.dim() == 3:
#                 if token.shape[1] != 1:
#                     raise ValueError(f"prom_list[{i}] must have shape [B,1,D] if 3D, got {token.shape}")
#                 token = token.squeeze(1)   # [B,1,D] -> [B,D]
#             elif token.dim() != 2:
#                 raise ValueError(f"prom_list[{i}] must be 2D or 3D, got {token.dim()}D")

#             prompts[:, self.x_positions[i], :] = token.to(
#                 device=prompts.device,
#                 dtype=prompts.dtype
#             )

#         return prompts

class Prompt_Cat5(nn.Module):
    def __init__(self, dataset_name, dtype, token_embedding):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            template = "A photo of a vehicle wearing X on top, X underneath, X shoes, having X hairstyle and carrying X."
        else:
            template = "A photo of a person wearing X on top , X underneath , X hairstyle , X shoes , carrying X ."
                            
        self.num_attributes = 5

        tokenized_prompts = clip.tokenize(template).to(token_embedding.weight.device)
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)

        self.tokenized_prompts = tokenized_prompts
        self.register_buffer("template_embedding", embedding)   # [1, 77, D]

        x_token_id = clip.tokenize("X")[0, 1].item()
        x_positions = (tokenized_prompts[0] == x_token_id).nonzero(as_tuple=False).view(-1)

        self.register_buffer("x_positions", x_positions)
        self.dtype = dtype

    def forward(self, prom_list):
        B = prom_list[0].shape[0]
        prompts = self.template_embedding.expand(B, -1, -1).clone()

        for i in range(self.num_attributes):
            prompts[:, self.x_positions[i], :] = prom_list[i].squeeze(1).type(self.dtype)

        return prompts


class Detailed_Prompt_Cat5(nn.Module):
    """
    Template:
    "A photo of a S person wearing X on top , X underneath , X hairstyle , X shoes , carrying X ."
    - tìm vị trí S riêng
    - tìm 5 vị trí X riêng
    """
    def __init__(self, dataset_name, dtype, token_embedding):
        super().__init__()

        if dataset_name == "VehicleID" or dataset_name == "veri":
            template = "A photo of a S vehicle wearing X on top , X underneath , X hairstyle , X shoes , carrying X ."
        else:
            template = "A photo of a S person wearing X on top , X underneath , X hairstyle , X shoes , carrying X ."

        self.num_attributes = 5
        self.dtype = dtype

        tokenized_prompts = clip.tokenize(template).to(token_embedding.weight.device)
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)

        self.tokenized_prompts = tokenized_prompts
        self.register_buffer("template_embedding", embedding)   # [1, 77, D]

        s_token_id = clip.tokenize("S")[0, 1].item()
        x_token_id = clip.tokenize("X")[0, 1].item()

        s_positions = (tokenized_prompts[0] == s_token_id).nonzero(as_tuple=False).view(-1)
        x_positions = (tokenized_prompts[0] == x_token_id).nonzero(as_tuple=False).view(-1)

        assert s_positions.numel() == 1, f"Expected 1 S position, got {s_positions.numel()} at {s_positions.tolist()}"
        assert x_positions.numel() == 5, f"Expected 5 X positions, got {x_positions.numel()} at {x_positions.tolist()}"

        self.register_buffer("s_position", s_positions)   # [1]
        self.register_buffer("x_positions", x_positions)  # [5]

    def forward(self, person_token, prom_list):
        """
        person_token: [B,1,D] or [B,D]
        prom_list order must follow template:
            [top_token, underneath_token, hairstyle_token, shoes_token, carrying_token]
        """
        if person_token.dim() == 2:
            person_token = person_token.unsqueeze(1)

        B = person_token.shape[0]
        prompts = self.template_embedding.expand(B, -1, -1).clone()

        prompts[:, self.s_position.item(), :] = person_token.squeeze(1).type(self.dtype)

        for i in range(self.num_attributes):
            token_i = prom_list[i]
            if token_i.dim() == 2:
                token_i = token_i.unsqueeze(1)
            prompts[:, self.x_positions[i], :] = token_i.squeeze(1).type(self.dtype)

        return prompts


# class DetailCrossBlock(nn.Module):
#     """
#     Một detail block gần với ý paper:
#     - q = current queries
#     - k,v = concat(queries, patch_tokens)
#     - cross attention
#     - refinement bằng 1 Transformer block
#     """
#     def __init__(self, embed_dim):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = embed_dim // 64

#         self.cross_attn = nn.MultiheadAttention(
#             self.embed_dim,
#             self.num_heads,
#             batch_first=True
#         )

#         # dùng 1 layer cho mỗi block; stack 6 block ở ngoài sẽ gần paper hơn
#         self.cross_modal_transformer = Transformer(
#             width=self.embed_dim,
#             layers=1,
#             heads=self.num_heads
#         )

#         self.ln_pre_t = LayerNorm(self.embed_dim)
#         self.ln_pre_i = LayerNorm(self.embed_dim)
#         self.ln_post = LayerNorm(self.embed_dim)

#         scale = self.cross_modal_transformer.width ** -0.5
#         proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
#         attn_std = scale
#         fc_std = (2 * self.cross_modal_transformer.width) ** -0.5

#         for block in self.cross_modal_transformer.resblocks:
#             nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
#             nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
#             nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
#             nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

#         nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
#         nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

#     def cross_former(self, q, k, v):
#         x = self.cross_attn(
#             self.ln_pre_t(q),
#             self.ln_pre_i(k),
#             self.ln_pre_i(v),
#             need_weights=False
#         )[0]  # [B, nq, D]

#         # residual với queries cũ
#         x = x + q

#         # NLD -> LND
#         x = x.permute(1, 0, 2)
#         x = self.cross_modal_transformer(x)
#         # LND -> NLD
#         x = x.permute(1, 0, 2)

#         x = self.ln_post(x)
#         return x

#     def forward(self, q, patch_tokens):
#         kv = torch.cat([q, patch_tokens], dim=1)  # [B, nq+Np, D]
#         q = self.cross_former(q, kv, kv)
#         return q

class DetailCrossBlock(nn.Module):
    """
    Một block theo paper:
    - Cross-attention: Q = queries, K = V = concat(queries, patch_tokens)
    - Feed-forward network (2 lớp Linear + ReLU)
    - Residual + LayerNorm
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = embed_dim // 64

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            self.embed_dim,
            self.num_heads,
            batch_first=True
        )
        self.ln_pre_q = LayerNorm(self.embed_dim)
        self.ln_pre_kv = LayerNorm(self.embed_dim)
        self.ln_post_attn = LayerNorm(self.embed_dim)

        # FFN: 2 lớp Linear với ReLU
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1)
        )
        self.ln_post_ffn = LayerNorm(self.embed_dim)

        self._init_weights()

    def _init_weights(self):
        scale = self.embed_dim ** -0.5
        attn_std = scale
        proj_std = scale
        nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
        nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, q, patch_tokens):
        """
        q: [B, n_queries, D]
        patch_tokens: [B, N_patch, D]
        """
        # Concatenate queries với patch tokens làm key/value
        kv = torch.cat([q, patch_tokens], dim=1)  # [B, n_queries+N_patch, D]

        # Cross-attention
        q_norm = self.ln_pre_q(q)
        kv_norm = self.ln_pre_kv(kv)
        attn_out, _ = self.cross_attn(q_norm, kv_norm, kv_norm, need_weights=False)
        q = q + attn_out                     # residual
        q = self.ln_post_attn(q)

        # FFN
        ffn_out = self.ffn(q)
        q = q + ffn_out                      # residual
        q = self.ln_post_ffn(q)

        return q

class DetailTokenBranch(nn.Module):
    """
    Một nhánh sinh 1 detail token:
    patch tokens -> learnable queries -> block_ca detail blocks -> avg pool -> IM2TEXT
    """
    def __init__(self, clip_model, n_querie, block_ca):
        super().__init__()
        self.embed_dim = clip_model.visual.output_dim
        self.n_querie = n_querie
        self.block_ca = block_ca

        scale = self.embed_dim ** -0.5
        self.queries = nn.Parameter(
            scale * torch.randn(1, self.n_querie, self.embed_dim)
        )

        self.blocks = nn.ModuleList([
            DetailCrossBlock(self.embed_dim) for _ in range(self.block_ca)
        ])

        self.mapper = IM2TEXT(
            embed_dim=clip_model.visual.output_dim,
            middle_dim=512,
            output_dim=clip_model.transformer.width,
            n_layer=3,
            dropout=0.1
        )

    def forward(self, patch_tokens):
        """
        patch_tokens: [B, Np, D]
        return: [B,1,Dt]
        """
        B = patch_tokens.shape[0]
        q = self.queries.expand(B, -1, -1)   # [B, n_querie, D]

        for blk in self.blocks:
            q = blk(q, patch_tokens)

        pooled = q.mean(dim=1)               # [B, D]
        token = self.mapper(pooled).unsqueeze(1)   # [B,1,Dt]
        return token
    
class Detailed_InversionPromptLearner5(nn.Module):
    """
    Input:
        image_features_for_inversion: [B, 1+N, D]
            - token 0: CLS
            - token 1..N: patch tokens
    """
    def __init__(self, cfg, clip_model):
        super().__init__()

        self.embed_dim = clip_model.visual.output_dim
        self.n_querie = cfg.MODEL.n_querie
        self.block_ca = cfg.MODEL.block_ca

        # S token từ CLS
        self.prompt_person = IM2TEXT(
            embed_dim=clip_model.visual.output_dim,
            middle_dim=512,
            output_dim=clip_model.transformer.width,
            n_layer=3,
            dropout=0.1
        )

        # 5 detail branches
        self.prompt_top = DetailTokenBranch(clip_model, self.n_querie, self.block_ca)
        self.prompt_underneath = DetailTokenBranch(clip_model, self.n_querie, self.block_ca)
        self.prompt_hairstyle = DetailTokenBranch(clip_model, self.n_querie, self.block_ca)
        self.prompt_shoes = DetailTokenBranch(clip_model, self.n_querie, self.block_ca)
        self.prompt_carrying = DetailTokenBranch(clip_model, self.n_querie, self.block_ca)

    def forward(self, image_features_for_inversion):
        """
        image_features_for_inversion: [B, 1+N, D]
        """
        cls_token = image_features_for_inversion[:, 0]      # [B, D]
        patch_tokens = image_features_for_inversion[:, 1:]  # [B, N, D]

        person_token = self.prompt_person(cls_token).unsqueeze(1)

        top_token = self.prompt_top(patch_tokens)
        underneath_token = self.prompt_underneath(patch_tokens)
        hairstyle_token = self.prompt_hairstyle(patch_tokens)
        shoes_token = self.prompt_shoes(patch_tokens)
        carrying_token = self.prompt_carrying(patch_tokens)

        return (
            person_token,
            top_token,
            underneath_token,
            hairstyle_token,
            shoes_token,
            carrying_token,
        )

class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE
        # ensure attribute exists even if SIE flags are disabled
        self.cv_embed = None
        self.att_flag = cfg.MODEL.ATT_FLAG
        self.s1_id_flag = cfg.SOLVER.STAGE1.S1_ID_LOSS_FLAG

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.text_classifier = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.text_classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.text_bottleneck = nn.BatchNorm1d(self.in_planes_proj)
        self.text_bottleneck.bias.requires_grad_(False)
        self.text_bottleneck.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        requested_device = str(getattr(cfg.MODEL, "DEVICE", "cuda")).lower()
        run_device = "cuda" if requested_device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        clip_model.to(run_device)

        self.image_encoder = clip_model.visual

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

        dataset_name = cfg.DATASETS.NAMES
        if self.att_flag == 3:
            self.prompt_learner = Prompt_Cat3(dataset_name, clip_model.dtype, clip_model.token_embedding)
            self.inversion_prompt_learner = InversionPromptLearner3(clip_model)
        elif self.att_flag == 5:
            # self.prompt_learner = Prompt_Cat5(dataset_name, clip_model.dtype, clip_model.token_embedding)
            # self.inversion_prompt_learner = InversionPromptLearner5(clip_model)
            self.prompt_learner = Detailed_Prompt_Cat5(dataset_name, clip_model.dtype, clip_model.token_embedding)
            self.inversion_prompt_learner = Detailed_InversionPromptLearner5(cfg, clip_model)
        else:
            raise ValueError(f"att_flag must be 3 or 5, but got {self.att_flag}")

        self.text_encoder = TextEncoder(clip_model)

    def forward(
        self,
        x=None,
        label=None,
        get_image=False,
        get_text=False,
        get_text_inversion=False,
        image_features_for_inversion = None,
        cam_label=None,
        view_label=None,
        prom_list=None
    ):
        if get_text is True:
            prompts = self.prompt_learner(label, prom_list)
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            return text_features

        if get_text_inversion == True:
            # Old flow (InversionPromptLearner3/5 + Prompt_Cat3/5):
            # prom_list = list(self.inversion_prompt_learner(image_features_for_inversion))
            # prompts = self.prompt_learner(prom_list)
            # text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            # return text_features
            person_token, top_token, underneath_token, hairstyle_token, shoes_token, carrying_token = \
                self.inversion_prompt_learner(image_features_for_inversion)
                
            prom_list = [top_token, underneath_token, hairstyle_token, shoes_token, carrying_token]
            prompts = self.prompt_learner(person_token, prom_list)
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            if self.s1_id_flag:
                text_feat = self.text_bottleneck(text_features)          # [B, 512]
                text_score = self.text_classifier(text_feat)
                return text_features, text_feat, text_score
            return text_features
        

        if get_image is True:
            if self.model_name == 'RN50':
                image_features_last, image_features, image_features_proj = self.image_encoder(x)
                return image_features_proj[0]
            elif self.model_name == 'ViT-B-16':
                if self.cv_embed is not None and cam_label is not None and view_label is not None:
                    cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
                elif self.cv_embed is not None and cam_label is not None:
                    cv_embed = self.sie_coe * self.cv_embed[cam_label]
                elif self.cv_embed is not None and view_label is not None:
                    cv_embed = self.sie_coe * self.cv_embed[view_label]
                else:
                    cv_embed = None
                image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed)
                return image_features_proj[:, 0], image_features_proj

        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            img_feature_last = nn.functional.avg_pool2d(
                image_features_last, image_features_last.shape[2:4]
            ).view(x.shape[0], -1)
            img_feature = nn.functional.avg_pool2d(
                image_features, image_features.shape[2:4]
            ).view(x.shape[0], -1)
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if self.cv_embed is not None and cam_label is not None and view_label is not None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif self.cv_embed is not None and cam_label is not None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif self.cv_embed is not None and view_label is not None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None

            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed)
            img_feature_last = image_features_last[:, 0]
            img_feature = image_features[:, 0]
            img_feature_proj = image_features_proj[:, 0]

        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)

        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj
        else:
            if self.neck_feat == 'after':
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature, img_feature_proj], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))