import unittest

import torch

from tools.profiling import (
    ThroughputMeter,
    count_parameters,
    estimate_mfu,
    estimate_transformer_flops,
    human_bytes,
    parameter_report,
)


class ProfilingTest(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(4, 3),
            torch.nn.Linear(3, 2),
        )

    def test_parameter_counts(self):
        self.assertEqual(count_parameters(self.model), 23)
        self.assertEqual(count_parameters(self.model, trainable_only=True), 23)

    def test_parameter_report_after_freezing(self):
        self.model[0].weight.requires_grad_(False)
        report = parameter_report(self.model)
        self.assertEqual(report["total"], 23)
        self.assertEqual(report["trainable"], 23 - 12)
        self.assertEqual(report["frozen"], 12)

    def test_throughput_meter(self):
        meter = ThroughputMeter().start()
        meter.update(100)
        stats = meter.stop()
        self.assertEqual(stats.tokens, 100)
        self.assertGreaterEqual(stats.elapsed_s, 0.0)
        self.assertGreaterEqual(stats.tokens_per_second, 0.0)
        self.assertEqual(stats.as_dict()["tokens"], 100)

    def test_flops_and_mfu(self):
        self.assertEqual(estimate_transformer_flops(100, 10), 6000)
        self.assertAlmostEqual(estimate_mfu(500, 1000), 0.5)
        self.assertEqual(human_bytes(1024), "1.0 KiB")


if __name__ == "__main__":
    unittest.main()
