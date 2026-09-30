import unittest


class TolokersRunnerContract(unittest.TestCase):
    def test_runner_resolves_only_weibo_or_tolokers_without_overwrite(self):
        from methods.bwgnn.src.run_formal import dataset_contract
        self.assertEqual(dataset_contract('weibo')['dataset_file'], 'datasets/weibo')
        contract = dataset_contract('tolokers')
        self.assertEqual(contract['dataset_file'], 'datasets/tolokers')
        self.assertEqual(contract['result_dataset'], 'tolokers')
        self.assertEqual(contract['expected_nodes'], 11758)

    def test_smoke_contract_is_bounded_to_five_epochs(self):
        from methods.bwgnn.src.run_formal import execution_contract
        self.assertEqual(execution_contract('smoke')['max_epoch'], 5)
        self.assertEqual(execution_contract('smoke')['run_type'], 'smoke')

    def test_amazon_contract_preserves_full_graph_and_uncovered_prefix(self):
        from methods.bwgnn.src.run_formal import dataset_contract
        contract = dataset_contract('amazon')
        self.assertEqual(contract['dataset_file'], 'datasets/amazon')
        self.assertEqual(contract['expected_nodes'], 11944)

    def test_tfinance_contract_resolves_frozen_full_graph(self):
        from methods.bwgnn.src.run_formal import dataset_contract
        self.assertEqual(dataset_contract('tfinance')['expected_nodes'], 39357)
