# OpenPI (Pi0.5) 的 PaliGemma Vision Encoder 结构分析（用于 FiLM/AdaLN 注入）

## 结论（可直接用于 Step A3）

在 OpenPI 的 `PI0Pytorch` 中，图像编码路径为：

```
PI0Pytorch
└── paligemma_with_expert: PaliGemmaWithExpertModel
    └── paligemma: PaliGemmaForConditionalGeneration
        └── model: PaliGemmaModel
            ├── vision_tower: SiglipVisionModel
            │   └── vision_model: SiglipVisionTransformer
            │       ├── embeddings: SiglipVisionEmbeddings
            │       ├── encoder: SiglipEncoder
            │       │   └── layers: ModuleList[SiglipEncoderLayer]
            │       └── post_layernorm: LayerNorm
            └── multi_modal_projector: PaliGemmaMultiModalProjector (Linear)
```

因此，**推荐的末端 N 个 vision blocks 获取方式**：

```python
vision_blocks = (
    pi.paligemma_with_expert
      .paligemma
      .model
      .vision_tower
      .vision_model
      .encoder
      .layers
)
film_blocks = vision_blocks[-num_film_blocks:]
```

> 注：这里的 `vision_tower` 由 `AutoModel.from_config(config.vision_config)` 构建；在 OpenPI 的 PaliGemma 配置下会解析为 `SiglipVisionModel`。

---

## 关键代码位置（以本仓库为准）

### 1) PI0Pytorch 如何调用 vision encoder

- 文件：`third_party/openpi/src/openpi/models_pytorch/pi0_pytorch.py`
  - `PI0Pytorch.embed_prefix(...)` 内对每个视角调用 `self.paligemma_with_expert.embed_image(img)`

- 文件：`third_party/openpi/src/openpi/models_pytorch/gemma_pytorch.py`
  - `PaliGemmaWithExpertModel.embed_image(image)` → `self.paligemma.model.get_image_features(image)`

### 2) PaliGemmaModel 的 image feature 计算

- 文件：`third_party/openpi/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py`
  - `PaliGemmaModel.__init__`：
    - `self.vision_tower = AutoModel.from_config(config=config.vision_config)`
    - `self.multi_modal_projector = PaliGemmaMultiModalProjector(config)`
  - `PaliGemmaModel.get_image_features(pixel_values)`：
    - `image_outputs = self.vision_tower(pixel_values)`
    - `selected_image_feature = image_outputs.last_hidden_state`（shape: `[B, N, D_vision]`）
    - `image_features = self.multi_modal_projector(selected_image_feature)`（shape: `[B, N, D_proj]`）

### 3) SigLIP Vision Encoder（vision blocks 的真实定义）

OpenPI 通过 `transformers_replace` 覆盖了 Transformers 中 SigLIP 的实现：

- 文件：`third_party/openpi/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py`
  - `class SiglipVisionModel`
    - `self.vision_model = SiglipVisionTransformer(config)`
  - `class SiglipVisionTransformer`
    - `self.embeddings = SiglipVisionEmbeddings(config)`
    - `self.encoder = SiglipEncoder(config)`
    - `self.post_layernorm = nn.LayerNorm(...)`
  - `class SiglipEncoder`
    - `self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])`

---

## 层结构与 Forward 签名（用于“手动 forward + 注入”）

### SiglipVisionModel.forward

```python
def forward(
    self,
    pixel_values,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    interpolate_pos_encoding: bool = False,
) -> BaseModelOutputWithPooling:
    return self.vision_model(...)
```

### SiglipVisionTransformer.forward（核心路径）

```python
hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=...)
encoder_outputs = self.encoder(inputs_embeds=hidden_states, attention_mask=None, ...)
last_hidden_state = self.post_layernorm(encoder_outputs.last_hidden_state)
```

> 注意：vision encoder 的 `attention_mask` 在这里默认是 `None`（与 text encoder 不同）。

### SiglipEncoderLayer.forward（每个 vision block）

```python
def forward(
    self,
    hidden_states: Tensor[B, N, D],
    attention_mask: Tensor | None,
    output_attentions: bool = False,
) -> tuple[Tensor, ...]:
    # LN → self_attn → residual → LN → MLP → residual
    return (hidden_states_out,)  # 以及可选 attn_weights
```

---

## FiLM/AdaLN 推荐注入点（不改 forward 签名）

在 `SiglipEncoder` 的 `layers` 循环中，对**末端 `num_film_blocks` 个 block 的输出**做调制：

```python
# 伪代码：在 adapter 里手动 forward
h = embeddings(pixel_values)  # [B,N,D_vision]
for i, block in enumerate(vision_blocks):
    h = block(h, attention_mask=None, output_attentions=False)[0]
    if i >= len(vision_blocks) - num_film_blocks:
        gamma, beta = film_generator(cond_emb)  # [B,D_vision], [B,D_vision]
        h = (1 + gamma[:, None, :]) * h + beta[:, None, :]
h = post_layernorm(h)
tokens = multi_modal_projector(h)  # [B,N,D_proj]
```

### 维度对齐提示

- FiLM 的通道维度 `D_vision` 应取 `vision_tower.config.hidden_size`
- `multi_modal_projector` 将 `D_vision → D_proj(=vision_config.projection_dim)`（在 OpenPI 配置里通常等于 text hidden size）

---

## 实施注意事项（后续 Step A3 会用到）

1. **dtype 逻辑**：`SiglipVisionTransformer.forward` 会在进入 encoder 前根据权重 dtype 做一次 cast（见实现）；手动 forward 时需要保持一致（建议跟随 patch_embedding / encoder 第一层 q_proj 的 dtype）。
2. **不侵入 third_party**：建议在 `src/vla_opt/adapters/openpi_pi05.py` 内实现 `encode_vision_with_film(...)`，不要直接改 `third_party/openpi`。

