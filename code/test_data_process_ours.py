import unittest
import os
import pickle
from GNN_utils.data_process_ours import get_batchs
from GNN_utils.mol_tree import Vocab

class TestGetBatchs(unittest.TestCase):
    def test_get_batchs(self):
        X = [('CCO', 0), ('CCN', 1), ('CCC', 0)]
        
        vocab = Vocab(['C', 'O', 'N'])
        data_path = './code/dataset/graph_data_ours'

        # id = get_batchs(X, vocab, data_path)

        self.assertTrue(os.path.exists(data_path))
        file_path = '.\code\dataset\graph_data_ours\CHEMBL203_val.p'
        with open(file_path, 'rb') as file:
            result = pickle.load(file)

        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)

        print(result)


if __name__ == '__main__':
    unittest.main()