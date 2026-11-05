"""Synthetic unit/integration controls; never empirical OPT-125M evidence."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from estimate_sensitivity_colab import (aggregate, digest, eligible_modules, eval_loss,
    fake_quantize_weight, measure_node, read_config, write_json)
from validate_sensitivity import validate, read


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.tensor([[.1, .3, .7, .2], [.4, .8, .6, .2],
                                                       [.9, .3, .4, .1], [.2, .7, .3, .8]]))

    def forward(self, input_ids, attention_mask, use_cache):
        return SimpleNamespace(logits=self.projection(torch.nn.functional.one_hot(input_ids, 4).float()))


def inputs():
    return {'input_ids': torch.tensor([[0, 1, 2, 3], [1, 3, 0, 0], [2, 1, 3, 0]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 0]])}


class NumericalTests(unittest.TestCase):
    def test_shift_and_batch_invariance(self):
        model = TinyLM()
        batch = inputs()
        a, raw = eval_loss(model, batch, 1, 'cpu')
        b, raw_b = eval_loss(model, batch, 3, 'cpu')
        self.assertEqual([r['target_count'] for r in raw], [3, 1, 2])
        self.assertAlmostEqual(a, b, places=6)
        with torch.no_grad():
            logits = model(**batch, use_cache=False).logits
            terms = [-torch.log_softmax(logits[i, t].float(), -1)[batch['input_ids'][i, t + 1]].item()
                     for i in range(3) for t in range(3) if batch['attention_mask'][i, t + 1]]
        self.assertAlmostEqual(a, sum(terms) / 6, places=7)
        self.assertEqual(sum(r['target_count'] for r in raw_b), 6)

    def test_zero_target_rejected(self):
        with self.assertRaises(ValueError):
            eval_loss(TinyLM(), {'input_ids': torch.tensor([[0, 0]]),
                      'attention_mask': torch.tensor([[1, 0]])}, 1, 'cpu')

    def test_quant_zero_tiny_and_identity(self):
        for bits in (2, 3, 4, 8):
            self.assertTrue(torch.equal(fake_quantize_weight(torch.zeros(4, dtype=torch.float16), bits), torch.zeros(4)))
            result = fake_quantize_weight(torch.tensor([1e-7, -1e-7], dtype=torch.float16), bits)
            self.assertTrue(torch.isfinite(result).all())
        x = torch.randn(10)
        y = fake_quantize_weight(x, 16)
        self.assertTrue(torch.equal(x, y))
        self.assertNotEqual(x.data_ptr(), y.data_ptr())

    def test_quant_levels_and_ties(self):
        x = torch.tensor([-1., -.5, 0., .5, 1.])
        self.assertTrue(torch.equal(fake_quantize_weight(x, 2), torch.tensor([-1., 0., 0., 0., 1.])))
        for bits in (2, 3, 4, 8):
            q = fake_quantize_weight(torch.linspace(-1, 1, 1001), bits)
            self.assertLessEqual(q.unique().numel(), 2 ** bits - 1)
            self.assertEqual(float(q.abs().max()), 1.)
        for bits in (-1, 0, 1, 5, 32):
            with self.assertRaises(ValueError):
                fake_quantize_weight(x, bits)
        with self.assertRaises(ValueError):
            fake_quantize_weight(torch.tensor([float('nan')]), 4)

    def test_restore_on_exception(self):
        model = TinyLM()
        weight = model.projection.weight.detach().clone()
        with patch('estimate_sensitivity_colab.eval_loss', side_effect=RuntimeError('injected failure')):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                measure_node(model, model.projection, 2, inputs(), 2, 'cpu')
        self.assertTrue(torch.equal(weight, model.projection.weight))

    def test_perturbation_order_and_identity(self):
        model = TinyLM()
        base = eval_loss(model, inputs(), 2, 'cpu')[0]
        self.assertEqual(base, measure_node(model, model.projection, 16, inputs(), 2, 'cpu')[0])
        forward = {b: measure_node(model, model.projection, b, inputs(), 2, 'cpu')[0] for b in (2, 3, 4, 8)}
        reverse = {b: measure_node(model, model.projection, b, inputs(), 2, 'cpu')[0] for b in (8, 4, 3, 2)}
        self.assertEqual(forward, reverse)
        self.assertEqual(base, eval_loss(model, inputs(), 2, 'cpu')[0])

    def test_tiny_opt_scope_and_tied_head(self):
        from transformers import OPTConfig, OPTForCausalLM
        model = OPTForCausalLM(OPTConfig(vocab_size=16, hidden_size=8, word_embed_proj_dim=8,
                    ffn_dim=16, num_hidden_layers=2, num_attention_heads=2))
        modules = eligible_modules(model)
        self.assertEqual(len(modules), 12)
        self.assertEqual(model.lm_head.weight.data_ptr(), model.model.decoder.embed_tokens.weight.data_ptr())
        self.assertFalse(any('lm_head' in n for n, _ in modules))
        model.model.decoder.layers[0].self_attn.k_proj.weight = model.model.decoder.layers[0].self_attn.q_proj.weight
        with self.assertRaisesRegex(ValueError, 'Shared'):
            eligible_modules(model)

    def test_tiny_opt_loss_matches_library(self):
        from transformers import OPTConfig, OPTForCausalLM
        torch.manual_seed(1)
        model = OPTForCausalLM(OPTConfig(vocab_size=4, hidden_size=8, word_embed_proj_dim=8,
                    ffn_dim=16, num_hidden_layers=1, num_attention_heads=2)).eval()
        batch = inputs()
        loss, _ = eval_loss(model, batch, 3, 'cpu')
        labels = batch['input_ids'].clone()
        labels[batch['attention_mask'] == 0] = -100
        with torch.no_grad():
            reference = model(**batch, labels=labels).loss.item()
        self.assertAlmostEqual(loss, reference, places=6)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def fixture(self):
        # Synthetic fixture generated ONLY inside a disposable unit-test directory.
        cfg = {'model': {'revision': 'a' * 40}, 'tokenizer': {'revision': 'b' * 40}, 'dataset': {'revision': 'c' * 40}, 'calibration': {'samples': 1, 'max_length': 2}, 'candidate_bits': [4],
               'checks': {'atol': 1e-4, 'rtol': 1e-4}}
        raw = [{'sample_index': 0, 'nll_sum': 1., 'target_count': 1}]
        row = {'module_name': 'model.decoder.layers.0.self_attn.q_proj', 'bits': 4, 'quantized_loss': 1., 'loss_delta': 0.,
               'per_sample': raw, 'sensitivity_rank': 1}
        data = {'schema_version': 1, 'code_commit': 'd' * 40, 'execution_status': 'completed', 'validation_status': 'pending', 'run_id': 'synthetic-test-only',
                'baseline_loss': 1., 'calibration_perplexity': math.e, 'baseline_per_sample': raw,
                'modules': [{'name': 'model.decoder.layers.0.self_attn.q_proj', 'shape': [768, 768], 'numel': 768 * 768}], 'candidate_bits': [4], 'rows': [row],
                'total_eligible': 72, 'partial_scope': True,
                'weights_before_sha256': 'a' * 64, 'weights_after_sha256': 'a' * 64,
                'controls': [{'name': n, 'loss': 1., 'passed': True} for n in
                             ('repeat_baseline', 'batch_size_one', 'identity_16bit', 'restored:model.decoder.layers.0.self_attn.q_proj')]}
        for name, value in {'effective_config.json': cfg, 'measurements.json': data,
                    'samples.json': {'row_indices': [0], 'texts': ['synthetic'], 'input_ids': [[0, 1]], 'attention_mask': [[1, 1]]},
                    'environment.json': {}, 'partial_rows.json': [row],
                    'status.json': {'execution_status': 'completed', 'run_id': 'synthetic-test-only'}}.items():
            write_json(self.root / name, value)
        self.rehash()

    def rehash(self):
        write_json(self.root / 'manifest.json', {'run_id': 'synthetic-test-only', 'files':
            {p.name: digest(p) for p in self.root.glob('*.json') if p.name not in ('manifest.json', 'status.json')}})

    def edit(self, change):
        p = self.root / 'measurements.json'
        d = json.loads(p.read_text())
        change(d)
        write_json(p, d)
        self.rehash()

    def test_valid_fixture_not_certified(self):
        self.fixture()
        report = validate(self.root)
        self.assertEqual(report['artifact_checks'], 'passed')
        self.assertEqual(report['scientific_validation'], 'pending')
        self.assertTrue(report['partial_scope'])

    def test_tampering_detected(self):
        self.fixture()
        (self.root / 'samples.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Hash mismatch'):
            validate(self.root)

    def test_missing_duplicate_wrong_counts_and_delta(self):
        changes = [lambda d: d.update(rows=[]), lambda d: d['rows'].append(copy.deepcopy(d['rows'][0])),
                   lambda d: d['rows'][0].update(loss_delta=99),
                   lambda d: d['rows'][0]['per_sample'][0].update(target_count=2),
                   lambda d: d.update(validation_status='validated'),
                   lambda d: d.update(partial_scope=False),
                   lambda d: d.update(controls=[])]
        for change in changes:
            with self.subTest(change=change):
                self.fixture()
                self.edit(change)
                with self.assertRaises(ValueError):
                    validate(self.root)

    def test_nonfinite_json_rejected(self):
        with self.assertRaises(ValueError):
            write_json(self.root / 'invalid.json', {'loss': float('nan')})
        with self.assertRaises(ValueError):
            aggregate([{'nll_sum': float('inf'), 'target_count': 1}])

    def test_unresolved_config_rejected(self):
        source = Path(__file__).resolve().parents[1] / 'configs/opt125m_calibration.yaml'
        with self.assertRaisesRegex(ValueError, 'immutable'):
            read_config(source)

    def test_json_overflow_and_duplicate_keys_rejected(self):
        path = self.root / 'bad.json'
        for content in ('{"loss": 1e999}', '{"loss": 1, "loss": 2}'):
            path.write_text(content)
            with self.assertRaises(ValueError):
                read(path)

    def test_launcher_stops_after_failed_checkout(self):
        import importlib.util
        source = Path(__file__).resolve().parents[1] / 'colab_stage3_opt125m.py'
        spec = importlib.util.spec_from_file_location('launcher', source)
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        calls = []
        def command(args, **kwargs):
            self.assertTrue(kwargs['check'])
            calls.append(args)
            if 'checkout' in args:
                raise launcher.subprocess.CalledProcessError(1, args)
        argv = ['launcher', '--commit', 'a' * 40, '--workdir', str(self.root / 'new')]
        with patch.object(sys, 'argv', argv), patch.object(launcher.subprocess, 'run', side_effect=command):
            with self.assertRaises(launcher.subprocess.CalledProcessError):
                launcher.main()
        self.assertEqual(len(calls), 2)


if __name__ == '__main__':
    unittest.main()
