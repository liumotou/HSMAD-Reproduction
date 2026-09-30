import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class CandidateContractTest(unittest.TestCase):
    def test_selected_original_gat_candidate_has_8x8_hidden_contract(self):
        from contracts import selected_candidate_contract

        contract = selected_candidate_contract()
        self.assertEqual(contract["hidden_heads"], 8)
        self.assertEqual(contract["hidden_features_per_head"], 8)
        self.assertEqual(contract["hidden_concat_dim"], 64)
        self.assertEqual(contract["hidden_activation"], "ELU")
        self.assertEqual(contract["output_heads"], 1)
        self.assertEqual(contract["output_dim"], 2)
        self.assertFalse(contract["output_concat"])
        self.assertEqual(contract["attention_negative_slope"], 0.2)


if __name__ == "__main__":
    unittest.main()
