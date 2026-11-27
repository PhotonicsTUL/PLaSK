import sys
import unittest

import numpy as np
import plask


class TensorFactoryOwnership(unittest.TestCase):

    def test_constructor_releases_array(self):
        for dtype in (float, complex):
            with self.subTest(dtype=dtype):
                source = np.arange(9, dtype=dtype).reshape(3, 3)
                if dtype is complex:
                    source += 1j
                references = sys.getrefcount(source)
                result = plask.tensor(source, dtype=dtype)
                self.assertEqual(sys.getrefcount(source), references)
                np.testing.assert_array_equal(np.asarray(result), source)

    def test_bad_shape_releases_array(self):
        source = np.ones((2, 3))
        references = sys.getrefcount(source)
        with self.assertRaises(TypeError):
            plask.tensor(source)
        self.assertEqual(sys.getrefcount(source), references)
