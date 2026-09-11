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
from ts_benchmark.baselines.nmask2.layers.LocalSummaryAttention import LocalSummaryAttention
from ts_benchmark.models.model_loader import get_models


class Nmask2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make_model(self, mode="rope", future=True, original=False, d_model=16, heads=2, **options):
        adapter_type = Nmask if original else Nmask2
        params = dict(seq_len=16, horizon=8, patch_len=4, stride=4,
                      d_model=d_model, d_ff=32, n_heads=heads, e_layers=2,
                      dropout=0.0, alpha=0.0, use_future_exog=future)
        if not original:
            params["channel_attn_mode"] = mode
            params["architecture"] = "joint"
        params.update(options)
        adapter = adapter_type(**params)
        adapter.config.enc_in = 5
        adapter.config.series_dim = 2
        adapter.config.criterion = torch.nn.L1Loss()
        return (NmaskModel if original else Nmask2Model)(adapter.config)

    def test_legacy_joint_matches_original_with_and_without_future_covariates(self):
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
            for key, value in {"architecture": "encoder_decoder", "covariate_layers": 1,
                               "channel_attn_type": "local_summary", "channel_window": 1,
                               "channel_summaries": 4, "temporal_attn_scope": "target_only"}.items():
                self.assertEqual(new_params.pop(key), value)
            self.assertEqual(new_params, old_params)
            self.assertEqual(after[after.index("--model-name") + 1], "nmask2.Nmask2")

    def test_local_summary_forward_backward_all_modes(self):
        for mode in ("rope", "none", "embedding"):
            for future in (False, True):
                with self.subTest(mode=mode, future=future):
                    model = self.make_model(mode, future, channel_attn_type="local_summary",
                                            channel_window=5, channel_summaries=2)
                    x, exog = torch.randn(2, 16, 5), torch.randn(2, 8, 3)
                    output, auxiliary = model(x, exog, None)
                    self.assertEqual(output.shape, (2, 8, 2))
                    (output.square().mean() + auxiliary).backward()
                    for layer in model.temporal_encoder.encoder_x.attn_layers:
                        self.assertTrue(layer.attention.use_rope)
                        self.assertIsNone(layer.c_attention)
                        attention = layer.local_channel_attention
                        self.assertGreater(attention.gate_query.weight.grad.abs().sum().item(), 0)
                        self.assertTrue(torch.isfinite(attention.key_projection.weight.grad).all())
                        if mode == "embedding":
                            self.assertTrue((attention.channel_embedding.grad.abs().sum(-1) > 0).all())

    def test_local_summary_matches_explicit_reference_at_boundaries(self):
        # Direct per-position/per-variable evaluation checks sparse gathering,
        # boundary masking, summary count clamping and both normalization stages.
        for mode, window, summaries in (("none", 5, 2), ("embedding", 1, 0), ("rope", 5, 8)):
            with self.subTest(mode=mode, window=window, summaries=summaries):
                module = LocalSummaryAttention(8, 2, 4, 2, window, summaries, mode,
                                               local_time_rope=False).double()
                x = torch.randn(2, 4, 3, 8, dtype=torch.float64)
                qk = x if module.channel_embedding is None else x + module.channel_embedding
                q = module.query_projection(qk).reshape(2, 4, 3, 2, 4)
                k = module.key_projection(qk).reshape(2, 4, 3, 2, 4)
                v = module.value_projection(x).reshape(2, 4, 3, 2, 4)
                if module.rope is not None:
                    q, k = module._rotate_channels(q), module._rotate_channels(k)
                count = min(summaries, 3)
                if count:
                    sk = module._pool(k[:, 2:].permute(0, 1, 3, 2, 4), count)
                    sv = module._pool(v[:, 2:].permute(0, 1, 3, 2, 4), count)
                expected = torch.empty(2, 2, 3, 8, dtype=torch.float64)
                for batch in range(2):
                    for target in range(2):
                        for position in range(3):
                            contexts = []
                            for variable in range(2, 4):
                                start, stop = max(0, position - window // 2), min(3, position + window // 2 + 1)
                                keys = k[batch, variable, start:stop].transpose(0, 1)
                                values = v[batch, variable, start:stop].transpose(0, 1)
                                if count:
                                    keys = torch.cat([keys, sk[batch, variable - 2]], dim=1)
                                    values = torch.cat([values, sv[batch, variable - 2]], dim=1)
                                scores = (q[batch, target, position, :, None] * keys).sum(-1) / 2
                                context = (scores.softmax(-1).unsqueeze(-1) * values).sum(1).flatten()
                                contexts.append(context)
                            contexts = torch.stack(contexts)
                            gate_keys = contexts
                            if module.channel_embedding is not None:
                                identities = torch.nn.functional.linear(module.channel_embedding[0, 2:, 0], module.key_projection.weight)
                                gate_keys = gate_keys + identities
                            gate = (module.gate_query(qk[batch, target, position]) * gate_keys).sum(-1) / (8 ** 0.5)
                            expected[batch, target, position] = module.out_projection((gate.softmax(-1)[:, None] * contexts).sum(0))
                torch.testing.assert_close(module(x), expected, rtol=1e-9, atol=1e-9)

    def test_locality_global_summary_and_variable_permutation(self):
        torch.manual_seed(3)
        module = LocalSummaryAttention(8, 2, 4, 1, window=3, summaries=0, mode="none").eval()
        x = torch.randn(2, 4, 9, 8)
        changed = x.clone()
        changed[:, 1:, -1] += 10
        with torch.no_grad():
            original = module(x)
            local = module(changed)
            torch.testing.assert_close(original[:, :, 0], local[:, :, 0], rtol=0, atol=0)
            permuted = module(x[:, [0, 3, 1, 2]])
            torch.testing.assert_close(original, permuted)
            module.summaries = 1
            self.assertGreater((module(x)[:, :, 0] - module(changed)[:, :, 0]).abs().max().item(), 1e-5)

    def test_local_summary_short_sequence_checkpoint_and_batching(self):
        module = LocalSummaryAttention(8, 2, 3, 1, window=5, summaries=4, mode="embedding").eval()
        restored = LocalSummaryAttention(8, 2, 3, 1, window=5, summaries=4, mode="embedding").eval()
        restored.load_state_dict(module.state_dict())
        x = torch.randn(3, 3, 1, 8)
        with torch.no_grad():
            expected = module(x)
            self.assertTrue(torch.isfinite(expected).all())
            torch.testing.assert_close(expected, restored(x), rtol=0, atol=0)
            torch.testing.assert_close(expected[:1], restored(x[:1]))

    def test_local_summary_invalid_options(self):
        for options in ({"channel_attn_type": "bad"}, {"channel_window": 0},
                        {"channel_window": 4}, {"channel_window": True},
                        {"channel_summaries": -1}, {"channel_summaries": 1.5}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                Nmask2(seq_len=16, **options)
        with self.assertRaisesRegex(ValueError, "covariate"):
            LocalSummaryAttention(8, 2, 1, 1)

    def test_target_only_temporal_attention_routing_and_gradients(self):
        for kind in ("full", "local_summary"):
            for mode in ("rope", "none", "embedding"):
                for future in (False, True):
                    with self.subTest(kind=kind, mode=mode, future=future):
                        model = self.make_model(mode, future, channel_attn_type=kind,
                                                temporal_attn_scope="target_only")
                        handles, captured = [], []
                        for layer in model.temporal_encoder.encoder_x.attn_layers:
                            handles.append(layer.attention.register_forward_pre_hook(
                                lambda m, args: captured.append(args[0].shape)))
                        try:
                            output, auxiliary = model(torch.randn(3, 16, 5), torch.randn(3, 8, 3), None)
                            self.assertEqual(output.shape, (3, 8, 2))
                            (output.square().mean() + auxiliary).backward()
                        finally:
                            for handle in handles:
                                handle.remove()
                        self.assertEqual(len(captured), 2)
                        self.assertTrue(all(shape[0] == 3 * 2 for shape in captured))
                        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                                            if p.grad is not None))
                        for layer in model.temporal_encoder.encoder_x.attn_layers:
                            self.assertGreater(layer.attention.query_projection.weight.grad.abs().sum().item(), 0)

    def test_target_only_covariates_skip_temporal_mixing(self):
        model = self.make_model("embedding", channel_attn_type="local_summary",
                                temporal_attn_scope="target_only").eval()
        layer = model.temporal_encoder.encoder_x.attn_layers[0]
        x = torch.randn(2, 5, 7, 16)
        captured = []
        handle = layer.local_channel_attention.register_forward_pre_hook(
            lambda m, args: captured.append(args[0].detach().clone()))
        with torch.no_grad():
            original, _ = layer(x.reshape(-1, 7, 16), 5)
            changed = x.clone()
            changed[:, 2:, -1] += torch.randn_like(changed[:, 2:, -1]) * 10
            perturbed, _ = layer(changed.reshape(-1, 7, 16), 5)
        handle.remove()
        # Covariates only pass token-wise normalization before channel attention.
        torch.testing.assert_close(captured[0][:, 2:], layer.norm1(x[:, 2:]))
        # Their later FFNs/norms cannot spread a perturbation to other patches.
        torch.testing.assert_close(original.reshape(2, 5, 7, 16)[:, 2:, :-1],
                                   perturbed.reshape(2, 5, 7, 16)[:, 2:, :-1], rtol=0, atol=0)
        # Parameter layout is unchanged, allowing checkpoint reuse in either mode.
        baseline = self.make_model("embedding", channel_attn_type="local_summary")
        baseline.load_state_dict(model.state_dict())
        self.assertEqual(Nmask2(seq_len=16, architecture="joint").config.temporal_attn_scope, "all")
        with self.assertRaisesRegex(ValueError, "temporal_attn_scope"):
            Nmask2(seq_len=16, temporal_attn_scope="bad")

    def test_local_time_rope_relative_positions_and_odd_tail(self):
        module = LocalSummaryAttention(10, 2, 3, 1, mode="none")
        x = torch.randn(2, 3, 2, 7, 5, dtype=torch.float64)
        rotated = module._rotate_time(x)
        torch.testing.assert_close(rotated.square().sum(-1), x.square().sum(-1))
        torch.testing.assert_close(rotated[..., -1], x[..., -1], rtol=0, atol=0)
        # A common shift of Q/K positions must preserve their dot product.
        padded = torch.cat((torch.zeros_like(x[..., :2, :]), x), dim=-2)
        shifted = module._rotate_time(padded)[..., 2:, :]
        torch.testing.assert_close(
            (rotated[..., 1, :] * rotated[..., 4, :]).sum(-1),
            (shifted[..., 1, :] * shifted[..., 4, :]).sum(-1), rtol=1e-12, atol=1e-12)

    def test_local_time_rope_scores_independent_of_channel_mode(self):
        for mode in ("none", "embedding", "rope"):
            with self.subTest(mode=mode):
                model = self.make_model(mode, channel_attn_type="local_summary", local_time_rope=False)
                self.assertFalse(model.temporal_encoder.encoder_x.attn_layers[0].local_channel_attention.local_time_rope)
                module = LocalSummaryAttention(8, 2, 3, 1, window=3, summaries=1, mode=mode).double()
                x = torch.randn(2, 3, 5, 8, dtype=torch.float64)
                qk = x if module.channel_embedding is None else x + module.channel_embedding
                q = module.query_projection(qk).reshape(2, 3, 5, 2, 4)
                k = module.key_projection(qk).reshape(2, 3, 5, 2, 4)
                if module.rope is not None:
                    q, k = module._rotate_channels(q), module._rotate_channels(k)
                captured = []
                handle = module.dropout.register_forward_pre_hook(lambda m, args: captured.append(args[0]))
                module(x).square().mean().backward()
                handle.remove()
                # Query p=2: check three local keys and the unchanged summary.
                query = q[:, 0, 2]
                logits = []
                for pos in (1, 2, 3):
                    key = k[:, 1, pos]
                    angle = torch.tensor([pos - 2, (pos - 2) / 100], dtype=torch.float64)
                    pairs = key.reshape(2, 2, 2, 2)
                    turned = torch.stack((pairs[..., 0] * angle.cos() - pairs[..., 1] * angle.sin(),
                                          pairs[..., 0] * angle.sin() + pairs[..., 1] * angle.cos()), -1).flatten(-2)
                    logits.append((query * turned).sum(-1) / 2)
                logits.append((query * k[:, 1].mean(1)).sum(-1) / 2)
                expected = torch.stack(logits, -1).softmax(-1)
                torch.testing.assert_close(captured[0][:, 0, :, 0, 2], expected, rtol=1e-10, atol=1e-10)
                self.assertTrue(torch.isfinite(module.key_projection.weight.grad).all())


class Nmask2EncoderDecoderTests(unittest.TestCase):
    make_model = Nmask2Tests.make_model

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def new_model(self, **options):
        return self.make_model(architecture="encoder_decoder", **options)

    def test_new_architecture_is_default_and_validates_options(self):
        adapter = Nmask2(seq_len=16, horizon=8)
        self.assertEqual(adapter.config.architecture, "encoder_decoder")
        self.assertEqual(adapter.config.temporal_attn_scope, "target_only")
        self.assertEqual(adapter.config.channel_attn_type, "local_summary")
        self.assertEqual(adapter.config.channel_window, 1)
        self.assertEqual(adapter.config.covariate_layers, 1)
        for options in ({"architecture": "bad"}, {"covariate_layers": 0},
                        {"covariate_layers": True}, {"covariate_layers": 1.5},
                        {"channel_attn_type": "full"}, {"temporal_attn_scope": "all"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                Nmask2(seq_len=16, **options)

    def test_read_only_memory_reused_and_no_target_to_covariate_path(self):
        model = self.new_model(mode="embedding").eval()
        core = model.temporal_encoder
        self.assertFalse(hasattr(core, "encoder_x"))
        encoded, inputs, handles = [], [], []
        handles.append(core.covariate_encoder.register_forward_hook(
            lambda module, args, result: encoded.append(result[0])))
        for layer in core.target_decoder.layers:
            handles.append(layer.register_forward_pre_hook(
                lambda module, args: inputs.append((args[0].shape, args[1], args[1].detach().clone()))))
        x, future = torch.randn(2, 16, 5), torch.randn(2, 8, 3)
        try:
            prediction, _ = model(x, future, None)
            self.assertEqual(len(encoded), 1)
            self.assertEqual(len(inputs), 2)
            for shape, memory, snapshot in inputs:
                self.assertEqual(shape[:2], (2, 2))
                self.assertEqual(memory.shape[:2], (2, 3))
                self.assertEqual(memory.data_ptr(), encoded[0].data_ptr())
                torch.testing.assert_close(memory, snapshot, rtol=0, atol=0)
            encoded[0].retain_grad()
            prediction.square().mean().backward()
            self.assertGreater(encoded[0].grad.abs().sum().item(), 0)
            changed = x.clone()
            changed[:, :, :2] = torch.randn_like(changed[:, :, :2]) * 3
            with torch.no_grad():
                model(changed, future, None)
            torch.testing.assert_close(encoded[0], encoded[1], rtol=0, atol=0)
        finally:
            for handle in handles:
                handle.remove()

    def test_gradients_all_modes_with_and_without_known_future(self):
        for mode in ("rope", "none", "embedding"):
            for future in (False, True):
                with self.subTest(mode=mode, future=future):
                    model = self.new_model(mode=mode, future=future, alpha=0.2,
                                           covariate_layers=2, d_model=12, heads=4)
                    x = torch.randn(2, 16, 5)
                    exog = torch.randn(2, 8, 3, requires_grad=True)
                    output, auxiliary = model(x, exog, None)
                    self.assertEqual(output.shape, (2, 8, 2))
                    self.assertTrue(torch.isfinite(output).all())
                    (output.square().mean() + auxiliary).backward()
                    core = model.temporal_encoder
                    for layer in core.covariate_encoder.attn_layers:
                        grad = layer.attention.value_projection.weight.grad
                        self.assertTrue(torch.isfinite(grad).all())
                        self.assertGreater(grad.abs().sum().item(), 0)
                    for layer in core.target_decoder.layers:
                        grad = layer.cross_attention.gate_query.weight.grad
                        self.assertGreater(grad.abs().sum().item(), 0)
                        if mode == "embedding":
                            self.assertGreater(layer.cross_attention.channel_embedding.grad.abs().sum().item(), 0)
                    if future:
                        self.assertGreater(exog.grad.abs().sum().item(), 0)

    def test_heads_short_horizons_and_window_boundaries(self):
        for method in ("future_patch", "all_history", "all_future", "all_sequence"):
            for window, summaries in ((1, 0), (5, 10)):
                with self.subTest(method=method, window=window, summaries=summaries):
                    model = self.new_model(predict_method=method, horizon=3,
                                           channel_window=window, channel_summaries=summaries)
                    result, _ = model(torch.randn(2, 16, 5), torch.randn(2, 3, 3), None)
                    self.assertEqual(result.shape, (2, 3, 2))
                    self.assertTrue(torch.isfinite(result).all())
                    result.square().mean().backward()

    def test_checkpoint_batch_independence_and_future_label_independence(self):
        model = self.new_model(mode="embedding").eval()
        restored = self.new_model(mode="embedding").eval()
        checkpoint = io.BytesIO()
        torch.save(model.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored.load_state_dict(torch.load(checkpoint, weights_only=True))
        x, exog = torch.randn(3, 16, 5), torch.randn(3, 8, 3)
        with torch.no_grad():
            expected, _ = model(x, exog, None)
            actual, _ = restored(x, exog, torch.randn(3, 8, 2))
            one, _ = restored(x[:1], exog[:1], None)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        torch.testing.assert_close(actual[:1], one, rtol=1e-5, atol=1e-6)

    def test_without_future_mode_does_not_read_future_inputs(self):
        model = self.new_model(future=False).eval()
        x = torch.randn(2, 16, 5)
        with torch.no_grad():
            absent, _ = model(x, None, None)
            supplied, _ = model(x, torch.randn(2, 8, 3), None)
        torch.testing.assert_close(absent, supplied, rtol=0, atol=0)


class Nmask2FusionTests(unittest.TestCase):
    make_model = Nmask2Tests.make_model

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_qk_identity_preserves_dot_with_all_identity_modes(self):
        for mode in ("none", "rope", "embedding"):
            dot = LocalSummaryAttention(8, 2, 4, 2, mode=mode).double()
            qk = LocalSummaryAttention(8, 2, 4, 2, mode=mode, channel_fusion_mode="qk").double()
            missing = qk.load_state_dict(dot.state_dict(), strict=False)
            self.assertEqual(missing.missing_keys, ["gate_key.weight"])
            self.assertEqual(missing.unexpected_keys, [])
            with torch.no_grad():
                qk.gate_key.weight.copy_(torch.eye(8, dtype=torch.float64))
            x = torch.randn(2, 4, 5, 8, dtype=torch.float64)
            torch.testing.assert_close(dot(x), qk(x), rtol=1e-12, atol=1e-12)

    def test_fusion_scores_match_reference_and_values_are_unprojected(self):
        # W=1/R=0 makes each per-variable context exactly its V projection,
        # so we can check the full fusion independently of local attention.
        for mode in ("qk", "mlp"):
            module = LocalSummaryAttention(8, 2, 5, 2, window=1, summaries=0,
                                           mode="none", channel_fusion_mode=mode).double()
            x = torch.randn(2, 5, 3, 8, dtype=torch.float64)
            contexts = module.value_projection(x[:, 2:]).permute(0, 2, 1, 3)
            contexts = contexts[:, None].expand(2, 2, 3, 3, 8)
            query = module.gate_query(x[:, :2]).unsqueeze(-2).expand_as(contexts)
            if mode == "qk":
                keys = torch.nn.functional.linear(contexts, module.gate_key.weight)
                scores = (query * keys).sum(-1) / (8 ** 0.5)
            else:
                first, _, last = module.gate_mlp
                hidden = torch.nn.functional.gelu(torch.nn.functional.linear(
                    torch.cat((query, contexts), dim=-1), first.weight, first.bias))
                scores = torch.nn.functional.linear(hidden, last.weight).squeeze(-1)
            expected = module.out_projection((scores.softmax(-1).unsqueeze(-1) * contexts).sum(-2))
            actual = module(x)
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
            permuted = x[:, [0, 1, 4, 2, 3]]
            torch.testing.assert_close(module(permuted), actual, rtol=1e-12, atol=1e-12)

    def test_new_modes_model_gradients_checkpoint_and_validation(self):
        for architecture in ("encoder_decoder", "joint"):
            for fusion in ("qk", "mlp"):
                model = self.make_model(mode="embedding", architecture=architecture,
                                        channel_attn_type="local_summary", channel_fusion_mode=fusion)
                x, future = torch.randn(2, 16, 5), torch.randn(2, 8, 3)
                result, _ = model(x, future, None)
                result.square().mean().backward()
                modules = [module for module in model.modules() if isinstance(module, LocalSummaryAttention)]
                for module in modules:
                    self.assertEqual(module.channel_fusion_mode, fusion)
                    scorer = module.gate_key if fusion == "qk" else module.gate_mlp
                    for param in scorer.parameters():
                        self.assertIsNotNone(param.grad)
                        self.assertTrue(torch.isfinite(param.grad).all())
                        self.assertGreater(param.grad.abs().sum().item(), 0)
                    self.assertGreater(module.gate_query.weight.grad.abs().sum().item(), 0)
                restored = self.make_model(mode="embedding", architecture=architecture,
                                           channel_attn_type="local_summary", channel_fusion_mode=fusion)
                restored.load_state_dict(model.state_dict())
                model.eval()
                restored.eval()
                with torch.no_grad():
                    torch.testing.assert_close(model(x, future, None)[0], restored(x, future, None)[0], rtol=0, atol=0)
        self.assertEqual(Nmask2(seq_len=16).config.channel_fusion_mode, "dot")
        with self.assertRaisesRegex(ValueError, "channel_fusion_mode"):
            Nmask2(seq_len=16, channel_fusion_mode="invalid")
        with self.assertRaisesRegex(ValueError, "local_summary"):
            Nmask2(seq_len=16, architecture="joint", channel_fusion_mode="mlp")


if __name__ == "__main__":
    unittest.main()
