import importlib.util
from pathlib import Path
import unittest

import crowsetta

HERE = Path(__file__).parent
DATA_DIR = HERE.joinpath('test_data')

yarden2annot_module = HERE.joinpath('../src/gardner/yarden2annot.py')


class TestYarden2Annot(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(yarden2annot_module.name.replace('.py', ''),
                                                      yarden2annot_module)
        self.yarden2annot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.yarden2annot)

        self.test_mat = DATA_DIR.joinpath('mat').joinpath('llb3_annot_subset.mat')

    def test_yarden2annot(self):
        annot_list = self.yarden2annot.yarden2annot(annot_path=self.test_mat)
        self.assertTrue(
            type(annot_list) == list
        )
        self.assertTrue(
            all([type(el) == crowsetta.Annotation for el in annot_list])
        )


if __name__ == '__main__':
    unittest.main()
