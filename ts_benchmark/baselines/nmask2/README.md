# Nmask2

Registered as `nmask2.Nmask2`. The default architecture is now an independent
covariate encoder followed by a target decoder. Original `nmask` is unchanged.

## Architecture

1. Normalize and patch historical targets and historical/known future covariates.
   Retain the original shared patch embedding, target future placeholders and
   prediction heads to isolate the architecture change.
2. At every covariate depth, run temporal attention and retain its normalized
   output as `M_i` before channel attention and the FFN. Channel attention and
   the FFN form the transition into the next temporal layer. This encoder never
   receives target states.
3. Target block `i` consumes the previous target state `H_(i-1)` and `M_i`, then
   applies target temporal attention, per-covariate local-summary cross-attention,
   variable fusion, and a target FFN.
4. Decoder blocks read but do not update their corresponding covariate memory.
   Memory is not detached: prediction gradients train every covariate layer.
   Each target block has its own cross-attention projections.
5. Decode future target patches using the existing prediction head.

Historical and future target patches both query memory. No true future target
values are used as input. At least one target and one covariate are required.

## Configuration

Set `share_temporal_attn=true` to share the complete temporal-attention module
(Q/K/V and output projections, with RoPE) between target and covariate streams
at the same depth. The default is `false`. Different depths, LayerNorms and
FFNs remain independent; streams still run separately and keep the same memory
tap points. If covariate-calendar conditioning is enabled, its temporal
attention uses the same module too. Calendar channels still skip temporal
attention when `calendar_temporal_attn=false`. This switch requires
`architecture=encoder_decoder`. Restore checkpoints with the same switch value
used during training to preserve the intended parameter sharing.

```json
{
  "architecture": "encoder_decoder",
  "e_layers": 2,
  "channel_attn_type": "local_summary",
  "temporal_attn_scope": "target_only",
  "channel_window": 1,
  "channel_summaries": 4,
  "channel_group_gating": true,
  "channel_group_logit_bias": false,
  "local_time_rope": true,
  "channel_attn_mode": "rope",
  "covariate_self_channel_attn": false,
  "use_patch_mask_embedding": false,
  "use_future_exog": true
}
```

`e_layers` jointly controls covariate encoder and target decoder depth and must
be a positive integer. New architecture defaults are W=1, R=4.
`channel_attn_type=local_summary` and
`temporal_attn_scope=target_only` are required for `encoder_decoder`; incompatible
explicit values raise an error rather than silently changing the architecture.
`covariate_layers` has been removed and passing it now raises an error. Existing
encoder-decoder checkpoints created with a separate covariate depth must be
retrained with the unified layer structure.

Set `covariate_self_channel_attn=true` to mix external variables inside each
patch between consecutive covariate time-attention layers. At depth `i`, the
model first computes and saves `M_i` from time attention for Target Layer `i`,
then applies standard multi-head self-attention across covariates followed by
the layer FFN. That post-FFN state is the input to covariate time-attention
layer `i+1`. At the final depth, the post-FFN state is sent to the existing
covariate auxiliary prediction head instead. The auxiliary loss is active when
`use_future_exog=false` and future covariates are supplied as labels; with
known future covariates (`use_future_exog=true`), that loss remains disabled to
avoid reconstructing values already present in the input. Consequently there are
`e_layers` channel-attention and FFN blocks. The channel blocks inherit
`channel_attn_mode`: channel RoPE, no identity encoding, or learned variable
embeddings. Neither channel attention nor the FFN overwrites the saved
per-depth memories.

Set `use_patch_mask_embedding=true` to add an availability embedding to every
value patch. A shared `Linear(patch_len, d_model)` projects the binary mask and
its output is added to the patch representation. Historical observations,
known future covariates and calendar values use 1; future targets, unavailable
future covariates and right-padding positions use 0. This applies to both
architectures and defaults to false for checkpoint compatibility.

Each target patch attends separately to W local patches and R adaptive-average
summaries **per covariate**. A target-dependent softmax across covariates then
fuses their outputs. W is any positive integer counting the current patch and
W-1 preceding patches: target patch p reads [p-W+1, ..., p]. For W=3 this is
[p-2, p-1, p], never p+1. Even window sizes are supported. This is relative to
each query patch, not restricted to the pre-forecast historical segment.
Only local retrieval changes; summaries still pool the full covariate timeline.
R is nonnegative, with R=0 disabling summaries. Invalid boundary positions are
masked, and the summary count is capped at the number of memory patches.
The covariate encoder has a global temporal receptive field, so W limits direct
read locations, not the model's entire temporal receptive field.

Set `channel_group_gating=true` to replace the first-stage joint softmax over
local and summary slots with three source groups. The current patch contributes
directly. The preceding `W-1` patches and the `R` Exo-Summaries are normalized
within their own groups, then added through sample-dependent scalar gates:

```text
context = current + sigmoid(history_gate) * history
                  + sigmoid(summary_gate) * summary
```

Each gate is a shared `Linear([target, current, group], 1)`. Its weight starts
at zero and its bias at -3, so both optional contributions start at about
`sigmoid(-3) = 0.047`. `W=1` disables the history contribution and `R=0`
disables the summary contribution. Calendar channels still contribute only
their current patch. The second-stage fusion across covariates is unchanged.
The option defaults to true. Disabled models contain no group-gate parameters,
which preserves the earlier checkpoint layout.

As an alternative, set `channel_group_logit_bias=true` to retain one joint
softmax while adding learned group priors to its logits:

```text
current_logits' = current_logits
history_logits' = history_logits + b_h - log(W - 1)
summary_logits' = summary_logits + b_s - log(R)
```

`b_h` and `b_s` are learned scalars initialized to `-3`. The logarithmic terms
keep each group's initial total unnormalized mass approximately independent of
the number of tokens in that group. When fewer than `R` summaries are available,
the implementation uses the actual summary count. This option and
`channel_group_gating` cannot be enabled together.

| `channel_attn_mode` | Channel RoPE | Variable embeddings |
| --- | --- | --- |
| `rope` (default) | Enabled | No |
| `none` | Disabled | No |
| `embedding` | Disabled | Added to Q/K and variable-selection scores, never V |

Variable order is targets first, then covariates; keep it consistent during
training and prediction. Temporal self-attention retains the original RoPE.
`local_time_rope=true` independently rotates local Q/K using patch positions;
summary Q/K, values and the variable fusion gate do not receive that additional
rotation. These options apply independently of channel identity mode.

`use_future_exog=false` retains the historical-only ablation: learned future
covariate placeholders replace known values, and the covariate encoder feeds
the existing auxiliary forecast head. Supplied future covariate values may
supervise that auxiliary head during training, but are never forward inputs.

## Running

### Calendar covariates

Set `"use_calendar_exog": true` to append each existing time-mark feature as
an extra covariate channel. Default is false, preserving previous checkpoints
and predictions. Hourly inputs use hour-of-day, day-of-week, day-of-month and
day-of-year; other frequencies use the data loader's corresponding features.
This does not add public-holiday data. No CSV modification is needed.
With calendar channels enabled, nmask2 preserves the full inferred sampling
frequency (for example `10min` or `2h`) in training and future timestamp
generation, rather than reducing it to a unit or single-character alias.

The adapter passes historical and future time marks into the model. Only the
last `pred_len` target marks are used for future patches; preceding label marks
are ignored. Calendar features retain their fixed scaling rather than per-window
normalization. They are appended after ordinary covariates and use the shared
patch embedding. With `calendar_temporal_attn=true` (default), they pass through
the existing covariate temporal encoder. Setting it to false keeps calendar
patches out of that encoder while ordinary covariates are still encoded in time.
The false mode requires `architecture=encoder_decoder`.

In local-summary channel interaction, calendar channels always use W=1/R=0:
only the same-index local slot is allowed, with neighboring and summary slots
masked out. Ordinary covariates retain `channel_window`/`channel_summaries`.
Calendar and ordinary contexts then participate together in the selected
dot/qk/mlp/cross_attn fusion. This rule restricts direct channel-attention reads;
the temporal encoder can still mix calendar information across patches when
`calendar_temporal_attn=true`. When it is false, each calendar memory patch
remains position-local.

Set `covariate_calendar_attn=true` to let every ordinary covariate query the
calendar channels before the resulting covariate memory is exposed to the
target decoder. Each covariate layer follows the target layer order: temporal
self-attention, local-summary calendar attention, then its FFN. Calendar keys
remain read-only and restricted to W=1/R=0; ordinary covariates are updated,
while calendar channels are not. Its depth also equals `e_layers`. This option
defaults to false, requires `use_calendar_exog=true`, at
least one ordinary covariate, and `architecture=encoder_decoder`.
Its channel-attention projection modules are automatically shared with the
corresponding target decoder layer. Temporal attention, FFNs, channel layouts,
masks and variable embeddings remain stream-specific. No extra hyperparameter
is needed.

Calendar timestamps remain known when `use_future_exog=false`; they are not
replaced by future placeholders or included in auxiliary covariate prediction
loss. Calendar-only conditioning is supported. Both encoder_decoder and joint
architectures support this option with `channel_attn_type=local_summary`.
Enabling it changes the channel count, so retrain using matching configuration
and data frequency. The base data/scaler channel counts remain unchanged.

### Variable fusion modes

`channel_fusion_mode` selects the second-stage softmax over covariates after
each variable's local/summary attention. It works in both architectures when
`channel_attn_type=local_summary` and is independent of RoPE and variable identity.

| Mode | Score for target query q and per-variable representation z |
| --- | --- |
| `dot` (default) | `q = gate_query(target); score = dot(q, z) / sqrt(width)` |
| `qk` | `score = dot(q, gate_key(z)) / sqrt(width)` |
| `mlp` | `score = Linear(GELU(Linear(concat(q, z))))` |
| `cross_attn` | Standard multi-head cross-attention, with `n_heads` heads and Q/K/V/output projections |

When variable embeddings are enabled, their existing contribution is added to
z before scoring (including before the new key projection). In dot/qk/mlp modes,
values remain the original per-variable context. In cross_attn mode, a learned
V projection transforms that context without the variable identity addition.
All modes softmax over variables only.

The MLP is shared across variables, target channels and patches within each
layer. Its hidden size is `min(64, n_heads * head_dim)` and its output is one
scalar per variable. Different decoder layers have independent scorers.
QK mode adds a bias-free `width -> width` key projection. No extra V projection
is added in qk mode. Default dot mode preserves the existing parameter layout and behavior.
Train the new modes separately; their additional parameters require matching
checkpoint configurations.

`cross_attn` replaces the second-stage scalar gate with standard multi-head
cross-attention. Each target patch is one Q token; its per-covariate contexts
are the K/V tokens. The head count equals `n_heads`, with the same `head_dim`
as local attention (including its existing dimension rounding). Each head
uses `softmax(Q K^T / sqrt(head_dim))` over covariates, attention dropout, and
projected V. Concatenated heads pass through the existing output projection.
There is no second scalar fusion gate. The first-stage local/summary attention
is unchanged. Set `"channel_fusion_mode": "cross_attn"` to enable it in either
architecture with local_summary; train it with matching checkpoint settings.

For example, add `"channel_fusion_mode": "qk"` or
`"channel_fusion_mode": "mlp"` to `--model-hyper-params` and use a separate
`--save-path` for each experiment. Existing scripts continue to default to dot.

Run from the repository root in the environment used for nmask:

```bash
bash scripts/covariate_forecasting/nmask2.sh
bash scripts/covariate_forecasting/nmask2_local_summary.sh
bash scripts/train/nmask2.sh
```

The first script contains 24 fixed configurations using channel RoPE. The second
uses variable embeddings. Both now explicitly select the new architecture with
W=1 and R=4. Each configuration uses its dataset-specific `e_layers` value for
both covariate encoding and target decoding; learning rates, patch sizes and
forecast horizons are preserved.

The training search script uses variable embeddings and retains its existing
25 combinations per enabled case and enabled/commented dataset selection.
Results go to `<dataset>/nmask2/encoder_decoder_embedding_w1_r4/search`.
Fixed runs use `encoder_decoder_rope_w1_r4` or
`encoder_decoder_embedding_w1_r4` directories, separate from older results.

## Legacy baseline and checkpoints

Set `architecture=joint` to reproduce the previous shared-stack architecture.
For this mode, omitted settings default to `channel_attn_type=full`,
`temporal_attn_scope=all`, and W=5. It also supports the previous local-summary
and target-only ablations. Old checkpoints require `architecture=joint` and
matching original options; they cannot be loaded directly into the new model.
Train `encoder_decoder` from scratch. Architectural differences alone do not
establish that forecasting accuracy is unchanged.

## Validation

```bash
python -m unittest discover -s tests -p 'test_nmask2.py' -v
```

Tests cover memory reuse/immutability, independence from target inputs, gradient
flow into the covariate encoder, channel modes, short horizons, prediction heads,
checkpoint round trips, batch independence, future-label independence, and the
legacy architecture's equivalence to original nmask.
