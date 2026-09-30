import unittest
from methods.gin.src.run import build_run_spec
class GINRunnerContractTest(unittest.TestCase):
 def test_candidate_spec(self):
  s=build_run_spec({'dataset':'weibo','seed':0,'run_type':'smoke','result_dir':'results/experiments/gin/weibo/gin_h64_candidate/smoke/seed_0','max_epoch':5,'patience':50,'_config_sha256':'x'})
  self.assertEqual(s.result_label,'candidate_protocol_not_author_exact')
if __name__=='__main__':unittest.main()
