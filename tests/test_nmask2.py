import io
import json
from pathlib import Path
import shlex
import unittest

import torch

from ts_benchmark.baselines.nmask import Nmask
from ts_benchmark.baselines.nmask.models.nmask_model import NmaskModel
from ts_benchmark.baselines.nmask2 import Nmask2
from ts_benchmark.baselines.nmask2.models.nmask2_model import Nmask2Model
from ts_benchmark.models.model_loader import get_models


class Nmask2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make_model(self, mode="rope", future=True, original=False, d_model=16, heads=2):
        adapter_type = Nmask if original else Nmask2
        params = dict(seq_len=16, horizon=8, patch_len=4, stride=4,
                      d_model=d_model, d_ff=32, n_heads=heads, e_layers=2,
                      dropout=0.0, alpha=0.0, use_future_exog=future)
        if not original:
            params["channel_attn_mode"] = mode
        adapter = adapter_type(**params)
        adapter.config.enc_in = 5
        adapter.config.series_dim = 2
        adapter.config.criterion = torch.nn.L1Loss()
        return (NmaskModel if original else Nmask2Model)(adapter.config)

    def test_default_matches_original_with_and_without_future_covariates(self):
        for future in (False, True):
            with self.subTest(future=future):
                torch.manual_seed(19)
                original = self.make_model(original=True, future=future).eval()
                torch.manual_seed(19)
                copied = self.make_model(future=future).eval()
                self.assertEqual(original.state_dict().keys(), copied.state_dict().keys())
                for key in original.state_dict():
                    torch.testing.assert_close(original.state_dict()[key], copied.state_dict()[key], rtol=0, atol=0)
                x, exog = torch.randn(3, 16, 5), torch.randn(3, 8, 3)
                with torch.no_grad():
                    a, _ = original(x, exog, None)
                    b, _ = copied(x, exog, None)
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_modes_forward_backward_and_embedding_gradients(self):
        for mode in ("rope", "none", "embedding"):
            for future in (False, True):
                with self.subTest(mode=mode, future=future):
                    model = self.make_model(mode, future).train()
                    for layer in model.temporal_encoder.encoder_x.attn_layers:
                        self.assertTrue(layer.attention.use_rope)
                        self.assertEqual(layer.c_attention.use_rope, mode == "rope")
                        self.assertEqual(layer.channel_embedding is not None, mode == "embedding")
                    x, exog = torch.randn(3, 16, 5), torch.randn(3, 8, 3)
                    result, auxiliary = model(x, exog, None)
                    self.assertEqual(result.shape, (3, 8, 2))
                    self.assertTrue(torch.isfinite(result).all())
                    (result.square().mean() + auxiliary).backward()
                    for layer in model.temporal_encoder.encoder_x.attn_layers:
                        if mode == "embedding":
                            grad = layer.channel_embedding.grad
                            self.assertIsNotNone(grad)
                            self.assertTrue(torch.isfinite(grad).all())
                            self.assertGreater(grad.abs().sum().item(), 0)

    def test_embedding_added_to_queries_and_keys_only(self):
        model = self.make_model("embedding").eval()
        layer = model.temporal_encoder.encoder_x.attn_layers[0]
        captured = []
        handle = layer.c_attention.register_forward_pre_hook(
            lambda module, args: captured.append(tuple(t.detach().clone() for t in args[:3]))
        )
        try:
            with torch.no_grad():
                model(torch.randn(2, 16, 5), torch.randn(2, 8, 3), None)
        finally:
            handle.remove()
        query, key, value = captured[0]
        torch.testing.assert_close(query, key)
        torch.testing.assert_close(query - value, layer.channel_embedding.expand_as(query))

    def test_embedding_checkpoint_roundtrip_and_batch_independence(self):
        model = self.make_model("embedding").eval()
        checkpoint = io.BytesIO()
        torch.save(model.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = self.make_model("embedding").eval()
        restored.load_state_dict(torch.load(checkpoint, weights_only=True))
        x, exog = torch.randn(3, 16, 5), torch.randn(3, 8, 3)
        with torch.no_grad():
            expected, _ = model(x, exog, None)
            actual, _ = restored(x, exog, None)
            one, _ = restored(x[:1], exog[:1], None)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        torch.testing.assert_close(actual[:1], one, rtol=1e-5, atol=1e-6)

    def test_modes_keep_projection_sizes_for_odd_head_dimensions(self):
        sizes = []
        for mode in ("rope", "none", "embedding"):
            model = self.make_model(mode, d_model=12, heads=4)
            layer = model.temporal_encoder.encoder_x.attn_layers[0]
            sizes.append(layer.c_attention.query_projection.weight.shape)
            result, _ = model(torch.randn(2, 16, 5), torch.randn(2, 8, 3), None)
            self.assertEqual(result.shape, (2, 8, 2))
        self.assertEqual(sizes[0], sizes[1])
        self.assertEqual(sizes[0], sizes[2])

    def test_factory_registration_and_invalid_mode(self):
        factories = get_models({"models": [{"model_name": "nmask2.Nmask2",
                                            "model_hyper_params": {"seq_len": 16, "horizon": 8, "norm": True,
                                                                   "channel_attn_mode": "embedding"}}],
                                "recommend_model_hyper_params": {}})
        self.assertIsInstance(factories[0](), Nmask2)
        self.assertEqual(factories[0]().model_name, "Nmask2")
        self.assertEqual(Nmask2(seq_len=16).config.channel_attn_mode, "rope")
        self.assertTrue(Nmask2(seq_len=16).config.use_future_exog)
        with self.assertRaisesRegex(ValueError, "channel_attn_mode"):
            Nmask2(seq_len=16, channel_attn_mode="invalid")

    def test_script_preserves_reference_experiments(self):
        root = Path(__file__).resolve().parents[1] / "scripts/covariate_forecasting"
        def commands(name):
            return [shlex.split(line) for line in (root / name).read_text().splitlines()
                    if line.startswith("python ")]
        original, copied = commands("nmask.sh"), commands("nmask2.sh")
        self.assertEqual(len(copied), 24)
        self.assertEqual(len(original), len(copied))
        for before, after in zip(original, copied):
            for flag in ("--config-path", "--data-name-list", "--strategy-args", "--gpus", "--num-workers", "--timeout"):
                self.assertEqual(before[before.index(flag) + 1], after[after.index(flag) + 1])
            old_params = json.loads(before[before.index("--model-hyper-params") + 1])
            new_params = json.loads(after[after.index("--model-hyper-params") + 1])
            self.assertEqual(new_params.pop("channel_attn_mode"), "rope")
            self.assertTrue(new_params.pop("use_future_exog"))
            self.assertEqual(new_params, old_params)
            self.assertEqual(after[after.index("--model-name") + 1], "nmask2.Nmask2")


if __name__ == "__main__":
    unittest.main()
