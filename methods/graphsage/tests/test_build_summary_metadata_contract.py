import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class BuildSummaryMetadataContractTest(unittest.TestCase):
    def test_metadata_is_loaded_from_the_formal_seed_snapshot_not_hard_coded_weibo(self):
        from build_summary import summary_metadata

        with tempfile.TemporaryDirectory() as temporary:
            formal_dir = Path(temporary)
            seed_dir = formal_dir / "seed_0"
            seed_dir.mkdir()
            (seed_dir / "config_snapshot.json").write_text(json.dumps({
                "method": "GraphSAGE-GADBench-h64",
                "dataset": "tfinance",
                "protocol_version": "graphsage_gadbench_h64_candidate_strict_determinism",
                "paper_target": {"f1_macro": 0.6855, "auroc": 0.7594},
            }), encoding="utf-8")

            self.assertEqual(summary_metadata(formal_dir), {
                "method": "GraphSAGE-GADBench-h64",
                "dataset": "tfinance",
                "protocol_version": "graphsage_gadbench_h64_candidate_strict_determinism",
                "paper_f1_macro": 0.6855,
                "paper_auroc": 0.7594,
            })


if __name__ == "__main__":
    unittest.main()
