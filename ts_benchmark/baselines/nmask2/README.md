# Nmask2

Independent copy of `nmask`, registered as `nmask2.Nmask2`. The original model
is unchanged. This version adds one model hyperparameter:

| `channel_attn_mode` | Channel RoPE | Learned variable embeddings |
| --- | --- | --- |
| `"rope"` (default) | Enabled, as in nmask | No |
| `"none"` | Disabled | No |
| `"embedding"` | Disabled | Yes, added to Q and K only |

Temporal attention retains the original RoPE in every mode. Projection sizes
are identical across modes. Each encoder layer in `embedding` mode learns a
`[1, enc_in, d_model]` embedding, initialized with standard deviation 0.02 and
shared over samples and patch positions. Values and residuals retain their
content features. Embeddings follow the model input order: target channels
first, then exogenous channels. Keep that variable order consistent between
training and prediction.

The default `rope` mode has the same parameter layout and forward behavior as
the original nmask when other settings, including `use_future_exog`, match.
Train the other modes separately for a meaningful ablation;
changing the mode after training is not an equivalent experiment. Checkpoints
from `embedding` mode include its additional embedding parameters.

## Running

From the repository root, in the environment used for nmask:

```bash
bash scripts/covariate_forecasting/nmask2.sh
```

The script contains the same 24 independent experiment commands as `nmask.sh`,
using `--model-name "nmask2.Nmask2"` and `"channel_attn_mode": "rope"`.
To compare modes, change that parameter to `"none"` or `"embedding"`, and
change the save-path suffix to the corresponding mode to separate results.
Other experiment parameters are preserved. `use_future_exog=True` is the
model default and is explicitly set in the script, matching the current
server nmask setting. For historical-only experiments, set it to `false`.
Keep this setting identical across all compared channel attention modes.
The new option affects the active `TC_EncDec.py` model path; copied experimental
encoder variants are retained for reference.

## Local patches plus global summaries

Set these model hyperparameters to replace each layer's channel attention:

```json
{
  "channel_attn_type": "local_summary",
  "channel_window": 5,
  "channel_summaries": 4,
  "local_time_rope": true,
  "channel_attn_mode": "embedding"
}
```

`channel_attn_type="full"` remains the default and preserves the original
same-patch channel self-attention. `channel_window` is W, a positive odd number
of centered neighboring patch positions (including the query position).
`channel_summaries` is R, a nonnegative count of ordered adaptive-average-pooled
summary tokens per covariate. R=0 disables summaries. When the sequence has
fewer than R patches, the summary count is capped at the sequence length.
Out-of-range window positions are masked; they do not wrap or duplicate edge
values in the attention distribution.

In `local_summary`, each target patch queries W local tokens plus R summaries
**separately for each covariate**, then a target-dependent softmax gate combines
the per-variable outputs. This operates on the existing concatenated history
and future patch sequence after temporal attention, including historical and
future target queries. Only target channels receive this cross-variable update;
covariates continue through their temporal and feed-forward layers. True future
target values are never inputs. With `use_future_exog=False`, the existing
learned future covariate placeholders are used instead of known future values.

The existing `channel_attn_mode` still applies: `rope` rotates Q/K along the
channel axis before local gathering/pooling, `none` adds no variable identity,
and `embedding` adds identity to Q/K and the variable-selection gate, never V.
Identity is also included in the variable-selection gate. The existing temporal
self-attention RoPE is unchanged.

`local_time_rope=true` (default) independently applies time RoPE to local Q/K
using their original indices in the concatenated history/future patch sequence,
before gathering windows. It is active for all three `channel_attn_mode` values;
`rope` combines channel rotation with time rotation, while `embedding` combines
variable identities with time rotation. V, summary Q/K and the fusion gate do
not receive this additional time rotation. Summary keys remain pooled from
keys before time rotation because summaries cover intervals, not single patches.
For odd head dimensions, the last component is left unrotated. No new trainable
parameters or checkpoint tensors are introduced. Set `local_time_rope=false`
to reproduce the previous local-summary behavior with existing checkpoints.
Existing local-summary scripts enable this through the default; use a separate
save path when comparing runs with time RoPE enabled versus disabled.

Set `temporal_attn_scope="target_only"` to run temporal self-attention only
on endogenous channels (all their historical and future patches). The default
`"all"` preserves the previous behavior. Covariates bypass temporal Q/K/V
projection, attention and its residual update; they still receive token-wise
LayerNorm and FFN processing. Channel attention, `channel_attn_mode`, and
`local_time_rope` are independent of this setting. With `local_summary`, the
covariates therefore preserve patch-local content until targets retrieve their
local tokens and pooled summaries. With `full`, channel attention still updates
all channels and can indirectly transfer temporally mixed target information.
The option introduces no new parameters and supports existing checkpoints.

For the target-only experiment, add `"temporal_attn_scope": "target_only"`
to the model hyperparameters and use a separate result directory, for example
`<dataset>/nmask2/local_summary_embedding_w5_r4_target_only/search`.

For S targets, E covariates and P total patches, attention score computation is
O(B S E P (W+R) D), without constructing a P-by-P cross-attention matrix.
This is cheaper than full cross-time target-to-covariate attention when W+R is
small compared with P, but is not necessarily cheaper than the original
same-patch channel attention. The model's temporal self-attention remains
quadratic in P, and its features already carry global temporal context.
Thus W limits the direct read locations of this module, not the complete
model's temporal receptive field. W is measured in patch indices, not timestamps.

`scripts/covariate_forecasting/nmask2_local_summary.sh` provides the same 24
experiments with `local_summary`, W=5, R=4 and variable embeddings enabled.
Results are separated under `<dataset>/nmask2/local_summary_embedding_w5_r4`.
The original `nmask2.sh` remains the full-channel-attention comparison script.

```bash
python -m unittest discover -s tests -p 'test_nmask2.py' -v
```
