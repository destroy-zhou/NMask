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
            for key, value in {"architecture": "encoder_decoder",
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
        self.assertFalse(hasattr(adapter.config, "covariate_layers"))
        self.assertTrue(adapter.config.calendar_temporal_attn)
        self.assertFalse(adapter.config.covariate_calendar_attn)
        self.assertFalse(adapter.config.use_patch_mask_embedding)
        for options in ({"architecture": "bad"}, {"covariate_layers": 1},
                        {"e_layers": 0}, {"e_layers": True}, {"e_layers": 1.5},
                        {"channel_attn_type": "full"}, {"temporal_attn_scope": "all"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                Nmask2(seq_len=16, **options)

    def test_layerwise_read_only_memories_and_no_target_to_covariate_path(self):
        model = self.new_model(mode="embedding").eval()
        core = model.temporal_encoder
        self.assertFalse(hasattr(core, "encoder_x"))
        encoded, inputs, target_outputs, handles = [], [], [], []
        for layer in core.covariate_encoder.attn_layers:
            handles.append(layer.register_forward_hook(
                lambda module, args, result: encoded.append(result[0])))
        for layer in core.target_decoder.layers:
            handles.append(layer.register_forward_pre_hook(
                lambda module, args: inputs.append((args[0], args[1], args[1].detach().clone()))))
            handles.append(layer.register_forward_hook(
                lambda module, args, result: target_outputs.append(result)))
        x, future = torch.randn(2, 16, 5), torch.randn(2, 8, 3)
        try:
            prediction, _ = model(x, future, None)
            self.assertEqual(len(encoded), 2)
            self.assertEqual(len(inputs), 2)
            for index, (target_state, memory, snapshot) in enumerate(inputs):
                self.assertEqual(target_state.shape[:2], (2, 2))
                self.assertEqual(memory.shape[:2], (2, 3))
                torch.testing.assert_close(
                    memory,
                    core.covariate_encoder.norm(encoded[index]).reshape_as(memory),
                    rtol=0, atol=0,
                )
                torch.testing.assert_close(memory, snapshot, rtol=0, atol=0)
                memory.retain_grad()
            self.assertNotEqual(inputs[0][1].data_ptr(), inputs[1][1].data_ptr())
            self.assertEqual(inputs[1][1].shape, inputs[0][1].shape)
            self.assertEqual(inputs[1][0].data_ptr(), target_outputs[0].data_ptr())
            prediction.square().mean().backward()
            for _, memory, _ in inputs:
                self.assertGreater(memory.grad.abs().sum().item(), 0)
            changed = x.clone()
            changed[:, :, :2] = torch.randn_like(changed[:, :, :2]) * 3
            with torch.no_grad():
                model(changed, future, None)
            for index in range(2):
                torch.testing.assert_close(encoded[index], encoded[index + 2], rtol=0, atol=0)
        finally:
            for handle in handles:
                handle.remove()

    def test_gradients_all_modes_with_and_without_known_future(self):
        for mode in ("rope", "none", "embedding"):
            for future in (False, True):
                with self.subTest(mode=mode, future=future):
                    model = self.new_model(mode=mode, future=future, alpha=0.2,
                                           d_model=12, heads=4)
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
                                           channel_window=window, channel_summaries=summaries,
                                           use_patch_mask_embedding=True)
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

    def test_cross_attention_matches_pytorch_multihead_attention(self):
        for mode in ("none", "embedding"):
            module = LocalSummaryAttention(8, 2, 5, 2, window=1, summaries=0,
                                           mode=mode, channel_fusion_mode="cross_attn").double()
            reference = torch.nn.MultiheadAttention(8, 2, batch_first=True).double()
            with torch.no_grad():
                reference.in_proj_weight.copy_(torch.cat([
                    module.gate_query.weight, module.gate_key.weight, module.gate_value.weight]))
                reference.in_proj_bias.copy_(torch.cat([
                    module.gate_query.bias, module.gate_key.bias, module.gate_value.bias]))
                reference.out_proj.load_state_dict(module.out_projection.state_dict())
            x = torch.randn(2, 5, 3, 8, dtype=torch.float64, requires_grad=True)
            context = module.value_projection(x[:, 2:]).permute(0, 2, 1, 3)
            context = context[:, None].expand(2, 2, 3, 3, 8)
            q_input, k_input = x[:, :2], context
            if mode == "embedding":
                q_input = q_input + module.channel_embedding[:, :2]
                identity = torch.nn.functional.linear(module.channel_embedding[:, 2:, 0], module.key_projection.weight)
                k_input = k_input + identity[:, None, None]
            expected, _ = reference(q_input.reshape(12, 1, 8),
                                    k_input.reshape(12, 3, 8), context.reshape(12, 3, 8),
                                    need_weights=False)
            expected = expected.reshape(2, 2, 3, 8)
            actual = module(x)
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
            actual_grad = torch.autograd.grad(actual.sum(), x, retain_graph=True)[0]
            expected_grad = torch.autograd.grad(expected.sum(), x)[0]
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-10, atol=1e-10)

    def test_cross_attention_architectures_heads_gradients_and_checkpoints(self):
        for architecture in ("encoder_decoder", "joint"):
            for mode in ("none", "rope", "embedding"):
                options = dict(mode=mode, architecture=architecture, d_model=12, heads=4,
                               channel_attn_type="local_summary", channel_fusion_mode="cross_attn")
                model = self.make_model(**options)
                x, future = torch.randn(2, 16, 5), torch.randn(2, 8, 3)
                result, _ = model(x, future, None)
                self.assertEqual(result.shape, (2, 8, 2))
                result.square().mean().backward()
                for module in model.modules():
                    if not isinstance(module, LocalSummaryAttention):
                        continue
                    self.assertEqual(module.n_heads, 4)
                    for projection in (module.gate_query, module.gate_key, module.gate_value, module.out_projection):
                        self.assertTrue(torch.isfinite(projection.weight.grad).all())
                        self.assertGreater(projection.weight.grad.abs().sum().item(), 0)
                restored = self.make_model(**options)
                restored.load_state_dict(model.state_dict())
                model.eval()
                restored.eval()
                with torch.no_grad():
                    output = restored(x, future, None)[0]
                    torch.testing.assert_close(model(x, future, None)[0], output, rtol=0, atol=0)
                    torch.testing.assert_close(restored(x[:1], future[:1], None)[0], output[:1], rtol=1e-5, atol=1e-6)
        with self.assertRaisesRegex(ValueError, "local_summary"):
            Nmask2(seq_len=16, architecture="joint", channel_fusion_mode="cross_attn")

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


class Nmask2CalendarTests(unittest.TestCase):
    make_model = Nmask2Tests.make_model

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_calendar_masks_only_center_and_preserves_regular_attention(self):
        for mode in ("none", "rope", "embedding"):
            for fusion in ("dot", "qk", "mlp", "cross_attn"):
                module = LocalSummaryAttention(8, 2, 5, 1, window=5, summaries=3,
                                               mode=mode, channel_fusion_mode=fusion,
                                               calendar_channels=2).eval()
                regular = LocalSummaryAttention(8, 2, 5, 1, window=5, summaries=3,
                                                mode=mode, channel_fusion_mode=fusion).eval()
                regular.load_state_dict(module.state_dict())
                x = torch.randn(2, 5, 7, 8)
                captured, reference = [], []
                a = module.dropout.register_forward_pre_hook(lambda m, args: captured.append(args[0]))
                b = regular.dropout.register_forward_pre_hook(lambda m, args: reference.append(args[0]))
                try:
                    output = module(x)
                    regular(x)
                    weights = captured[0]
                    torch.testing.assert_close(weights[..., :2, :, :], reference[0][..., :2, :, :], rtol=0, atol=0)
                    calendar_weights = weights[..., -2:, :, :]
                    expected = torch.zeros_like(calendar_weights)
                    expected[..., 2] = 1
                    torch.testing.assert_close(calendar_weights, expected, rtol=0, atol=0)
                    changed = x.clone()
                    changed[:, -2:, :3] += torch.randn_like(changed[:, -2:, :3]) * 10
                    changed[:, -2:, 4:] += torch.randn_like(changed[:, -2:, 4:]) * 10
                    torch.testing.assert_close(module(changed)[:, :, 3], output[:, :, 3], rtol=0, atol=0)
                finally:
                    a.remove()
                    b.remove()

    def test_calendar_forward_backward_future_availability_and_alignment(self):
        for architecture in ("encoder_decoder", "joint"):
            for future in (True, False):
                model = self.make_model(architecture=architecture, future=future,
                                        use_calendar_exog=True, freq="h", alpha=0.2,
                                        channel_attn_type="local_summary", channel_fusion_mode="cross_attn")
                x, exog = torch.randn(2, 16, 5), torch.randn(2, 8, 3)
                past = torch.randn(2, 16, 4, requires_grad=True)
                marks = torch.randn(2, 12, 4, requires_grad=True)
                result, auxiliary = model(x, exog, None, past, marks)
                self.assertEqual(result.shape, (2, 8, 2))
                self.assertEqual(model.temporal_encoder.c_in, 9)
                (result.square().mean() + auxiliary).backward()
                self.assertGreater(past.grad.abs().sum().item(), 0)
                self.assertGreater(marks.grad[:, -8:].abs().sum().item(), 0)
                self.assertEqual(marks.grad[:, :4].abs().sum().item(), 0)
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                model.eval()
                other = marks.detach().clone()
                other[:, :4] += 100
                with torch.no_grad():
                    torch.testing.assert_close(model(x, exog, None, past, marks)[0],
                                               model(x, exog, None, past, other)[0], rtol=0, atol=0)

    def test_calendar_validation_and_checkpoint(self):
        options = dict(architecture="encoder_decoder", use_calendar_exog=True, freq="d")
        model, restored = self.make_model(**options).eval(), self.make_model(**options).eval()
        self.assertEqual(model.calendar_channels, 3)
        restored.load_state_dict(model.state_dict())
        x, exog = torch.randn(2, 16, 5), torch.randn(2, 8, 3)
        past, future = torch.randn(2, 16, 3), torch.randn(2, 8, 3)
        with torch.no_grad():
            expected = model(x, exog, None, past, future)[0]
            torch.testing.assert_close(expected, restored(x, exog, None, past, future)[0], rtol=0, atol=0)
            torch.testing.assert_close(expected[:1], restored(x[:1], exog[:1], None, past[:1], future[:1])[0])
        with self.assertRaisesRegex(ValueError, "time marks"):
            model(x, exog, None)
        with self.assertRaisesRegex(ValueError, "boolean"):
            Nmask2(seq_len=16, use_calendar_exog=1)
        with self.assertRaisesRegex(ValueError, "boolean"):
            Nmask2(seq_len=16, calendar_temporal_attn=0)
        with self.assertRaisesRegex(ValueError, "boolean"):
            Nmask2(seq_len=16, covariate_calendar_attn=1)
        with self.assertRaisesRegex(ValueError, "boolean"):
            Nmask2(seq_len=16, use_patch_mask_embedding=1)
        with self.assertRaisesRegex(ValueError, "use_calendar_exog"):
            Nmask2(seq_len=16, covariate_calendar_attn=True)
        with self.assertRaisesRegex(ValueError, "local_summary"):
            Nmask2(seq_len=16, architecture="joint", use_calendar_exog=True)
        with self.assertRaisesRegex(ValueError, "encoder_decoder"):
            Nmask2(seq_len=16, architecture="joint", channel_attn_type="local_summary",
                   use_calendar_exog=True, calendar_temporal_attn=False)
        with self.assertRaisesRegex(ValueError, "encoder_decoder"):
            Nmask2(seq_len=16, architecture="joint", channel_attn_type="local_summary",
                   use_calendar_exog=True, covariate_calendar_attn=True)

        adapter = Nmask2(
            seq_len=16, horizon=8, patch_len=4, stride=4, d_model=8,
            d_ff=16, n_heads=2, e_layers=1, dropout=0.0, alpha=0.0,
            use_calendar_exog=True, freq="h",
            covariate_calendar_attn=True,
        )
        adapter.config.enc_in = adapter.config.series_dim = 1
        adapter.config.criterion = torch.nn.L1Loss()
        with self.assertRaisesRegex(ValueError, "ordinary covariates"):
            Nmask2Model(adapter.config)

    def test_patch_mask_embedding_tracks_value_availability(self):
        x, exog = torch.randn(2, 16, 5), torch.randn(2, 8, 3)
        past, future_mark = torch.randn(2, 16, 4), torch.randn(2, 8, 4)
        history_mask = torch.cat((torch.ones(4, 4), torch.zeros(1, 4)))
        known_future_mask = torch.cat((torch.ones(2, 4), torch.zeros(1, 4)))
        unknown_future_mask = torch.zeros(3, 4)

        for known_future in (False, True):
            with self.subTest(known_future=known_future):
                model = self.make_model(
                    architecture="encoder_decoder", future=known_future,
                    use_calendar_exog=True, freq="h",
                    use_patch_mask_embedding=True,
                ).train()
                captured = []
                handle = model.temporal_encoder.patch_mask_embedding.register_forward_pre_hook(
                    lambda module, args: captured.append(args[0].detach().clone())
                )
                try:
                    prediction, auxiliary = model(x, exog, None, past, future_mark)
                finally:
                    handle.remove()

                expected_target = torch.cat((history_mask, unknown_future_mask))
                expected_regular = torch.cat((
                    history_mask,
                    known_future_mask if known_future else unknown_future_mask,
                ))
                expected_calendar = torch.cat((history_mask, known_future_mask))
                expected = torch.cat((
                    expected_target.repeat(2, 1, 1),
                    expected_regular.repeat(3, 1, 1),
                    expected_calendar.repeat(4, 1, 1),
                )).unsqueeze(0).repeat(2, 1, 1, 1)
                torch.testing.assert_close(captured[0], expected)

                (prediction.square().mean() + auxiliary).backward()
                embedding = model.temporal_encoder.patch_mask_embedding
                self.assertGreater(embedding.weight.grad.abs().sum().item(), 0)
                self.assertGreater(embedding.bias.grad.abs().sum().item(), 0)

                partial = model.temporal_encoder._patch_observation_mask(
                    1, 1, 6, True, x,
                )
                torch.testing.assert_close(
                    partial,
                    torch.tensor([[[[1., 1., 1., 1.], [1., 1., 0., 0.]]]]),
                )

        joint = self.make_model(
            mode="embedding", future=False, use_calendar_exog=False,
            channel_fusion_mode="dot", use_patch_mask_embedding=True,
        ).train()
        joint_prediction, joint_auxiliary = joint(x, exog, None)
        (joint_prediction.square().mean() + joint_auxiliary).backward()
        self.assertGreater(
            joint.temporal_encoder.patch_mask_embedding.weight.grad.abs().sum().item(),
            0,
        )

    def test_calendar_temporal_attention_can_be_bypassed_independently(self):
        x, exog = torch.randn(2, 16, 5), torch.randn(2, 8, 3)
        past, future = torch.randn(2, 16, 4), torch.randn(2, 8, 4)
        encoder_inputs, decoder_memories = {}, {}

        for enabled in (True, False):
            model = self.make_model(
                architecture="encoder_decoder", use_calendar_exog=True, freq="h",
                calendar_temporal_attn=enabled,
            ).eval()
            core = model.temporal_encoder
            handles = [
                core.covariate_encoder.attn_layers[0].register_forward_pre_hook(
                    lambda module, args, flag=enabled:
                        encoder_inputs.__setitem__(flag, args[0].detach().clone())),
                core.target_decoder.register_forward_pre_hook(
                    lambda module, args, flag=enabled:
                        decoder_memories.__setitem__(
                            flag, [item.detach().clone() for item in args[1]]
                        )),
            ]
            try:
                with torch.no_grad():
                    output, _ = model(x, exog, None, past, future)
                self.assertEqual(output.shape, (2, 8, 2))
            finally:
                for handle in handles:
                    handle.remove()

        # Enabled: all 3 ordinary + 4 calendar channels enter the encoder.
        self.assertEqual(encoder_inputs[True].shape[0], 2 * 7)
        # Disabled: only the 3 ordinary covariates enter it.
        self.assertEqual(encoder_inputs[False].shape[0], 2 * 3)

        bypassed_calendar = decoder_memories[False][0][:, -4:]
        disabled_model = self.make_model(
            architecture="encoder_decoder", use_calendar_exog=True, freq="h",
            calendar_temporal_attn=False,
        ).eval()
        disabled_model.load_state_dict(model.state_dict())
        with torch.no_grad():
            hist, _ = disabled_model.temporal_encoder.x_patch_embedding(past.transpose(1, 2))
            fut, _ = disabled_model.temporal_encoder.x_patch_embedding(future.transpose(1, 2))
        expected = torch.cat((hist, fut), dim=1).reshape_as(bypassed_calendar)
        torch.testing.assert_close(bypassed_calendar, expected, rtol=0, atol=0)

        train_model = self.make_model(
            architecture="encoder_decoder", use_calendar_exog=True, freq="h",
            calendar_temporal_attn=False,
        ).train()
        train_past, train_future = past.clone().requires_grad_(), future.clone().requires_grad_()
        prediction, _ = train_model(x, exog, None, train_past, train_future)
        prediction.square().mean().backward()
        self.assertGreater(train_past.grad.abs().sum().item(), 0)
        self.assertGreater(train_future.grad.abs().sum().item(), 0)
        encoder_grad = train_model.temporal_encoder.covariate_encoder.attn_layers[0].attention
        self.assertGreater(encoder_grad.value_projection.weight.grad.abs().sum().item(), 0)

    def test_covariate_calendar_attention_shares_projections_in_all_fusion_modes(self):
        for fusion in ("dot", "qk", "mlp", "cross_attn"):
            with self.subTest(fusion=fusion):
                model = self.make_model(
                    mode="embedding", architecture="encoder_decoder",
                    use_calendar_exog=True, freq="h",
                    covariate_calendar_attn=True,
                    channel_fusion_mode=fusion, e_layers=2,
                )
                core = model.temporal_encoder
                common = (
                    "query_projection", "key_projection", "value_projection",
                    "out_projection", "gate_query",
                )
                optional = {
                    "dot": (), "qk": ("gate_key",), "mlp": ("gate_mlp",),
                    "cross_attn": ("gate_key", "gate_value"),
                }[fusion]
                for index, layer in enumerate(core.covariate_calendar_decoder.layers):
                    source = core.target_decoder.layers[index].cross_attention
                    for name in common + optional:
                        self.assertIs(getattr(layer.cross_attention, name),
                                      getattr(source, name))
                parameters = list(model.parameters())
                self.assertEqual(len(parameters), len({id(parameter) for parameter in parameters}))

    def test_ordinary_covariates_can_attend_to_read_only_calendar(self):
        model = self.make_model(
            mode="embedding", architecture="encoder_decoder",
            use_calendar_exog=True, freq="h", calendar_temporal_attn=False,
            covariate_calendar_attn=True, channel_fusion_mode="cross_attn",
        ).train()
        core = model.temporal_encoder
        conditioner = core.covariate_calendar_decoder
        covariate_attention = conditioner.layers[0].cross_attention
        target_attention = core.target_decoder.layers[0].cross_attention
        for name in (
            "query_projection", "key_projection", "value_projection",
            "out_projection", "gate_query", "gate_key", "gate_value",
        ):
            self.assertIs(getattr(covariate_attention, name), getattr(target_attention, name))
        self.assertIsNot(covariate_attention.channel_embedding,
                         target_attention.channel_embedding)
        self.assertIsNot(conditioner.layers[0].linear1,
                         core.target_decoder.layers[0].linear1)
        inputs, decoder_memory = [], []
        handles = [
            conditioner.layers[0].register_forward_pre_hook(
                lambda module, args: inputs.append(
                    (args[0].detach().clone(), args[1].detach().clone())
                )
            ),
            core.target_decoder.register_forward_pre_hook(
                lambda module, args: decoder_memory.append(
                    [item.detach().clone() for item in args[1]]
                )
            ),
        ]
        x, exog = torch.randn(2, 16, 5), torch.randn(2, 8, 3)
        past = torch.randn(2, 16, 4, requires_grad=True)
        future = torch.randn(2, 8, 4, requires_grad=True)
        try:
            prediction, _ = model(x, exog, None, past, future)
        finally:
            for handle in handles:
                handle.remove()

        regular_input, calendar_input = inputs[0]
        self.assertEqual(regular_input.shape[:2], (2, 3))
        self.assertEqual(calendar_input.shape[:2], (2, 4))
        for layer_memory in decoder_memory[0]:
            torch.testing.assert_close(
                layer_memory[:, -4:], calendar_input, rtol=0, atol=0,
            )
        self.assertFalse(torch.equal(decoder_memory[0][0][:, :3], regular_input))

        prediction.square().mean().backward()
        self.assertGreater(past.grad.abs().sum().item(), 0)
        self.assertGreater(future.grad.abs().sum().item(), 0)
        self.assertEqual(len(conditioner.layers), len(core.covariate_encoder.attn_layers))
        layer = conditioner.layers[0]
        temporal_attention = layer.time_attention
        self.assertGreater(
            temporal_attention.value_projection.weight.grad.abs().sum().item(), 0,
        )
        attention = layer.cross_attention
        for projection in (
            attention.value_projection, attention.gate_query,
            attention.gate_key, attention.gate_value, attention.out_projection,
        ):
            self.assertIsNotNone(projection.weight.grad)
            self.assertGreater(projection.weight.grad.abs().sum().item(), 0)
        # The calendar-only local softmax has one slot per variable, while the
        # shared target path also trains this projection against ordinary exog.
        self.assertGreater(
            attention.query_projection.weight.grad.abs().sum().item(), 0,
        )

        restored = self.make_model(
            mode="embedding", architecture="encoder_decoder",
            use_calendar_exog=True, freq="h", calendar_temporal_attn=False,
            covariate_calendar_attn=True, channel_fusion_mode="cross_attn",
        ).eval()
        restored.load_state_dict(model.state_dict())
        model.eval()
        with torch.no_grad():
            expected = model(x, exog, None, past, future)[0]
            actual = restored(x, exog, None, past, future)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        with_temporal_calendar = self.make_model(
            architecture="encoder_decoder", use_calendar_exog=True, freq="h",
            calendar_temporal_attn=True, covariate_calendar_attn=True,
        ).eval()
        with torch.no_grad():
            output, _ = with_temporal_calendar(x, exog, None, past, future)
        self.assertEqual(output.shape, (2, 8, 2))
        self.assertTrue(torch.isfinite(output).all())

    def test_calendar_only_without_regular_covariates(self):
        for known_future in (True, False):
            for calendar_attn in (True, False):
                with self.subTest(known_future=known_future, calendar_attn=calendar_attn):
                    adapter = Nmask2(
                        seq_len=16, horizon=3, patch_len=4, stride=4,
                        d_model=8, d_ff=16, n_heads=2, e_layers=1, dropout=0.0,
                        use_calendar_exog=True, freq="h", use_future_exog=known_future,
                        calendar_temporal_attn=calendar_attn,
                        use_patch_mask_embedding=True,
                    )
                    adapter.config.enc_in = adapter.config.series_dim = 1
                    adapter.config.criterion = torch.nn.L1Loss()
                    model = Nmask2Model(adapter.config)
                    output, auxiliary = model(
                        torch.randn(2, 16, 1), None, None,
                        torch.randn(2, 16, 4), torch.randn(2, 3, 4),
                    )
                    self.assertEqual(output.shape, (2, 3, 1))
                    self.assertEqual(auxiliary, 0)
                    output.square().mean().backward()
                    self.assertGreater(
                        model.temporal_encoder.patch_mask_embedding.weight.grad.abs().sum().item(),
                        0,
                    )
                    self.assertTrue(torch.isfinite(output).all())

    def test_calendar_sampling_frequency_and_rolling_timestamp_alignment(self):
        import numpy as np
        import pandas as pd
        from ts_benchmark.baselines.utils import get_time_mark
        for freq in ("10min", "2h", "h", "D"):
            for columns in (1, 3):
                index = pd.date_range("2024-01-31 22:00", periods=40, freq=freq, name="date")
                frame = pd.DataFrame(np.zeros((40, columns)), index=index)
                adapter = Nmask2(seq_len=16, horizon=8, use_calendar_exog=True)
                tune = adapter.single_forecasting_hyper_param_tune if columns == 1 else adapter.multi_forecasting_hyper_param_tune
                tune(frame)
                self.assertEqual(pd.tseries.frequencies.to_offset(adapter.config.freq), pd.tseries.frequencies.to_offset(freq))
                stamps = np.stack((index[:16].to_numpy(), index[3:19].to_numpy()))
                actual = adapter._padding_time_stamp_mark(stamps, 8)
                expected_stamps = np.stack((index[:24].to_numpy(), index[3:27].to_numpy()))
                expected = get_time_mark(expected_stamps, 1, freq)
                np.testing.assert_array_equal(actual, expected)
                _, _, history_marks, future_marks = adapter._get_rolling_data(
                    np.zeros((2, 16, columns)), None, actual, 0)
                np.testing.assert_array_equal(history_marks, expected[:, :16])
                np.testing.assert_array_equal(future_marks[:, -8:], expected[:, 16:24])

    def test_calendar_patch_channel_order_and_no_window_normalization(self):
        model = self.make_model(architecture="encoder_decoder", use_calendar_exog=True, freq="h").eval()
        core = model.temporal_encoder
        captured = []
        handle = core.covariate_encoder.attn_layers[0].register_forward_pre_hook(
            lambda m, args: captured.append(args[0].detach().clone()))
        x, exog = torch.randn(2, 16, 5), torch.randn(2, 8, 3)
        # Distinct constants per sample/feature expose accidental flattening
        # across channel/batch boundaries and unwanted instance normalization.
        past = torch.arange(8, dtype=torch.float32).reshape(2, 1, 4).expand(2, 16, 4)
        future = past[:, :8] + 10
        try:
            with torch.no_grad():
                model(x, exog, None, past, future)
        finally:
            handle.remove()
        actual = captured[0].reshape(2, 7, 8, 16)[:, -4:]
        with torch.no_grad():
            hist, _ = core.x_patch_embedding(past.transpose(1, 2))
            fut, _ = core.x_patch_embedding(future.transpose(1, 2))
        expected = torch.cat((hist, fut), dim=1).reshape(2, 4, 8, 16)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
