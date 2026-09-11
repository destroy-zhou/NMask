# Nmask2

Registered as `nmask2.Nmask2`. The default architecture is now an independent
covariate encoder followed by a target decoder. Original `nmask` is unchanged.

## Architecture

1. Normalize and patch historical targets and historical/known future covariates.
   Retain the original shared patch embedding, target future placeholders and
   prediction heads to isolate the architecture change.
2. Encode each covariate along time using `covariate_layers` attention/FFN blocks.
   This encoder runs once per forward pass and never receives target states.
3. Each of the `e_layers` target decoder blocks applies target temporal attention,
   per-covariate local-summary cross-attention, variable fusion, and a target FFN.
4. All decoder blocks read the same covariate memory. They do not update or
   normalize it in place. Memory is not detached: prediction gradients train the
   covariate encoder normally. Each block has its own cross-attention projections.
5. Decode future target patches using the existing prediction head.

Historical and future target patches both query memory. No true future target
values are used as input. At least one target and one covariate are required.

## Configuration

```json
{
  "architecture": "encoder_decoder",
  "covariate_layers": 1,
  "e_layers": 2,
  "channel_attn_type": "local_summary",
  "temporal_attn_scope": "target_only",
  "channel_window": 1,
  "channel_summaries": 4,
  "local_time_rope": true,
  "channel_attn_mode": "rope",
  "use_future_exog": true
}
```

`e_layers` controls target decoder depth; `covariate_layers` controls the separate
covariate encoder depth (positive integer, default 1). New architecture defaults
are W=1, R=4. `channel_attn_type=local_summary` and
`temporal_attn_scope=target_only` are required for `encoder_decoder`; incompatible
explicit values raise an error rather than silently changing the architecture.

Each target patch attends separately to W local patches and R adaptive-average
summaries **per covariate**. A target-dependent softmax across covariates then
fuses their outputs. W is a positive odd number of centered patch positions;
R is nonnegative, with R=0 disabling summaries. Invalid boundary positions are
masked, and the summary count is capped at the number of memory patches.
The covariate encoder has a global temporal receptive field, so W limits direct
read locations, not the model's entire temporal receptive field.

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

Run from the repository root in the environment used for nmask:

```bash
bash scripts/covariate_forecasting/nmask2.sh
bash scripts/covariate_forecasting/nmask2_local_summary.sh
bash scripts/train/nmask2.sh
```

The first script contains 24 fixed configurations using channel RoPE. The second
uses variable embeddings. Both now explicitly select the new architecture with
W=1, R=4 and one covariate encoder layer. Dataset-specific decoder depths,
learning rates, patch sizes and forecast horizons are preserved.

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
