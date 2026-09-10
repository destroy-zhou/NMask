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
Identity is included in the gate because a constant variable-specific K offset
alone cancels in a softmax restricted to that variable. Temporal RoPE is unchanged.

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
