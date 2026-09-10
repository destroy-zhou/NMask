import json
from pathlib import Path
import shlex
import unittest

import numpy as np
import pandas as pd

from ts_benchmark.baselines.chronos2 import Chronos2
from ts_benchmark.models.model_loader import get_models


class Point:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class Pipeline:
    def predict_quantiles(self, tasks, **kwargs):
        self.tasks, self.kwargs = tasks, kwargs
        return [], [Point(np.repeat(task['target'][:, -1:], kwargs['prediction_length'], axis=1))
                    for task in tasks]


class AdapterTests(unittest.TestCase):
    def model(self, **kwargs):
        model = Chronos2(**kwargs)
        model.pipeline = Pipeline()
        return model

    def test_real_batch_maker_alignment_and_partial_batch(self):
        from ts_benchmark.evaluation.strategy.rolling_forecast import (
            RollingForecastEvalBatchMaker, RollingForecastPredictBatchMaker,
        )
        targets = pd.DataFrame({'y': np.arange(12, dtype=float)})
        exog = np.arange(24, dtype=float).reshape(12, 2) + 100
        maker = RollingForecastEvalBatchMaker(targets, [4, 6, 8], {'exog': exog})
        future = maker.make_batch_eval(2)['covariates']['exog']
        predictor = RollingForecastPredictBatchMaker(maker)
        model = self.model(seq_len=4, batch_size=2)
        first = model.batch_forecast(2, predictor, future, 0)
        np.testing.assert_array_equal(first[:, 0, 0], [3, 5])
        np.testing.assert_array_equal(model.pipeline.tasks[1]['future_covariates']['exog_0'], exog[6:8, 0])
        last = model.batch_forecast(2, predictor, future, 1)
        self.assertEqual(last.shape, (1, 2, 1))
        np.testing.assert_array_equal(model.pipeline.tasks[0]['target'], [[4, 5, 6, 7]])
        np.testing.assert_array_equal(model.pipeline.tasks[0]['past_covariates']['exog_1'], exog[4:8, 1])
        np.testing.assert_array_equal(model.pipeline.tasks[0]['future_covariates']['exog_1'], exog[8:10, 1])
        self.assertFalse(model.pipeline.kwargs['cross_learning'])
        self.assertFalse(predictor.has_more_batches())

    def test_multivariate_and_single_forecast(self):
        model = self.model(seq_len=4)
        series = pd.DataFrame(np.arange(20).reshape(10, 2))
        prediction = model.forecast(3, series)
        self.assertEqual(prediction.shape, (3, 2))
        np.testing.assert_array_equal(prediction, [[18, 19]] * 3)
        self.assertEqual(model.pipeline.tasks[0]['target'].shape, (2, 4))
        self.assertNotIn('future_covariates', model.pipeline.tasks[0])

    def test_reject_misaligned_future_covariates(self):
        model = self.model()
        with self.assertRaisesRegex(ValueError, 'Future covariates'):
            model._predict(np.ones((2, 4, 1)), np.ones((2, 4, 3)), np.ones((1, 2, 3)), 2)

    def test_model_factory(self):
        factories = get_models({'models': [{'model_name': 'chronos2.Chronos2',
                                            'model_hyper_params': {'seq_len': 168}}],
                                'recommend_model_hyper_params': {}})
        self.assertIsInstance(factories[0](), Chronos2)

    def test_reference_experiment_matrix(self):
        root = Path(__file__).resolve().parents[1]
        cases = []
        for line in (root / 'scripts/covariate_forecasting/nmask.sh').read_text().splitlines():
            if not line.startswith('python '):
                continue
            args = shlex.split(line)
            strategy = json.loads(args[args.index('--strategy-args') + 1])
            params = json.loads(args[args.index('--model-hyper-params') + 1])
            self.assertEqual(strategy['target_channel'], [-1])
            cases.append((args[args.index('--data-name-list') + 1], strategy['horizon'], params['seq_len']))
        expected = [(d + '.csv', h, s)
                    for d in 'NP PJM BE FR DE Energy Sdwpfm1 Sdwpfm2 Sdwpfh1 Sdwpfh2'.split()
                    for h, s in [(24, 168), (360, 720)]]
        expected += [(d + '.csv', h, s) for d in ['Colbun', 'Rapel'] for h, s in [(10, 60), (30, 180)]]
        self.assertEqual(cases, expected)


if __name__ == '__main__':
    unittest.main()
