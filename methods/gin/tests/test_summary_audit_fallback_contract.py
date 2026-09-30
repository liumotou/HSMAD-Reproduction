import unittest
import tempfile
from pathlib import Path

from methods.gin.audit.summarize import eligible_audit_path


class GINSummaryAuditFallbackContractTest(unittest.TestCase):
    def test_tolerance_audit_is_used_only_when_primary_is_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / 'audit_recompute').mkdir()
            (output / 'audit_recompute_tolerance_v1').mkdir()
            (output / 'audit_recompute/recompute_result.json').write_text(
                '{"status":"recompute_mismatch"}', encoding='utf-8'
            )
            (output / 'audit_recompute_tolerance_v1/recompute_result.json').write_text(
                '{"status":"recompute_match"}', encoding='utf-8'
            )
            self.assertEqual(
                eligible_audit_path(output).parent.name,
                'audit_recompute_tolerance_v1',
            )


if __name__ == '__main__':
    unittest.main()
