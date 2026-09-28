import csv
import tempfile
import unittest
from pathlib import Path

from methods.chebnet.src.run import append_run_row


class ChebNetRunLedgerContractTest(unittest.TestCase):
    def test_formal_ledger_appends_without_erasing_prior_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'runs.csv'
            fields = ['seed', 'status']
            append_run_row(path, fields, {'seed': 0, 'status': 'OK'})
            append_run_row(path, fields, {'seed': 1, 'status': 'OK'})
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows, [{'seed': '0', 'status': 'OK'}, {'seed': '1', 'status': 'OK'}])


if __name__ == '__main__':
    unittest.main()
