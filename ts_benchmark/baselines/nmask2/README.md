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
the original nmask. Train the other modes separately for a meaningful ablation;
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
All other parameters are preserved, including `use_future_exog=False` by
default. If testing known-future covariates, explicitly set
`"use_future_exog": true` consistently in every compared configuration.
The new option affects the active `TC_EncDec.py` model path; copied experimental
encoder variants are retained for reference.

```bash
python -m unittest discover -s tests -p 'test_nmask2.py' -v
```
