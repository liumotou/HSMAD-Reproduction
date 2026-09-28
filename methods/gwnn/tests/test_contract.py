import inspect
import unittest

import torch


class GWNNContractTests(unittest.TestCase):
    def test_paper_formula_wavelet_pair_is_inverse_without_sparsification(self):
        from methods.gwnn.src.model import build_paper_formula_wavelets

        edge_index = torch.tensor(
            [[0, 1, 1, 2, 0, 1, 2], [1, 0, 2, 1, 0, 1, 2]], dtype=torch.long
        )
        wavelet, inverse = build_paper_formula_wavelets(
            edge_index, num_nodes=3, scale=1.0, threshold=0.0, dtype=torch.float64
        )
        torch.testing.assert_close(
            wavelet @ inverse,
            torch.eye(3, dtype=torch.float64),
            rtol=1e-8,
            atol=1e-8,
        )

    def test_two_layer_model_shape_gradient_and_no_label_mask_arguments(self):
        from methods.gwnn.src.model import GWNNPaperFormulaCandidate

        signature = inspect.signature(GWNNPaperFormulaCandidate.forward)
        self.assertEqual(list(signature.parameters), ["self", "x", "wavelet", "inverse"])
        model = GWNNPaperFormulaCandidate(4, 64, 2, num_nodes=5, dropout=0.5)
        x = torch.randn(5, 4)
        support = torch.eye(5)
        logits = model(x, support, support)
        self.assertEqual(tuple(logits.shape), (5, 2))
        logits.sum().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_masked_protocol_isolates_train_validation_and_test(self):
        from methods.gwnn.src.protocol import (
            masked_cross_entropy,
            select_validation_threshold,
            test_metrics,
        )

        logits = torch.tensor(
            [[3.0, 0.0], [0.0, 3.0], [1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]],
            requires_grad=True,
        )
        labels = torch.tensor([0, 1, 0, 1, 0, 1])
        train = torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.bool)
        val = torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.bool)
        test = torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.bool)
        loss = masked_cross_entropy(logits, labels, train)
        changed = labels.clone(); changed[~train] = 1 - changed[~train]
        self.assertEqual(float(loss), float(masked_cross_entropy(logits, changed, train)))
        probabilities = torch.softmax(logits.detach(), dim=1)[:, 1]
        threshold, _ = select_validation_threshold(labels, probabilities, val)
        changed = labels.clone(); changed[test] = 1 - changed[test]
        self.assertEqual(threshold, select_validation_threshold(changed, probabilities, val)[0])
        original = test_metrics(labels, probabilities, test, threshold)
        changed = labels.clone(); changed[~test] = 1 - changed[~test]
        self.assertEqual(original, test_metrics(changed, probabilities, test, threshold))


if __name__ == "__main__":
    unittest.main()
