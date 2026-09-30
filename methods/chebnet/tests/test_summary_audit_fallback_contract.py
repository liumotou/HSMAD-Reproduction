import tempfile
import unittest
from pathlib import Path
from methods.chebnet.audit.summarize import eligible_audit_path

class ChebNetSummaryAuditFallbackContractTest(unittest.TestCase):
    def test_tolerant_audit_is_used_only_when_primary_is_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); (p/'audit_recompute').mkdir(); (p/'audit_recompute_tolerance_v1').mkdir()
            (p/'audit_recompute/recompute_result.json').write_text('{"status":"recompute_mismatch"}')
            (p/'audit_recompute_tolerance_v1/recompute_result.json').write_text('{"status":"recompute_match"}')
            self.assertEqual(eligible_audit_path(p).parent.name,'audit_recompute_tolerance_v1')

if __name__=='__main__': unittest.main()
