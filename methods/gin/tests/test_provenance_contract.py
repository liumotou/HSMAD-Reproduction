import unittest
from pathlib import Path
from methods.project_paths import project_root
class GINProvenanceContractTest(unittest.TestCase):
 def test_runner_hashes_own_method_sources(self):
  s=(project_root() / 'methods/gin/src/run.py').read_text()
  self.assertIn("methods/gin/src/model.py",s); self.assertNotIn("methods/chebnet/src/model.py",s)
if __name__=='__main__':unittest.main()
