import unittest
from pathlib import Path
class GINProvenanceContractTest(unittest.TestCase):
 def test_runner_hashes_own_method_sources(self):
  s=Path('/root/autodl-tmp/HSMAD/methods/gin/src/run.py').read_text()
  self.assertIn("methods/gin/src/model.py",s); self.assertNotIn("methods/chebnet/src/model.py",s)
if __name__=='__main__':unittest.main()
