import sys
import unittest

import numpy as np
import plask


class TensorArgumentOwnership(unittest.TestCase):

    def test_add_array_to_tensor(self):
        for dtype in (float, complex):
            for transpose in (False, True):
                with self.subTest(dtype=dtype, transpose=transpose):
                    source = np.arange(9, dtype=dtype).reshape(3, 3)
                    if dtype is complex:
                        source += 1j
                    if transpose:
                        source = source.T
                    result = plask.tensor(0., 0., 0., dtype=dtype)
                    references = sys.getrefcount(source)
                    result += source
                    self.assertEqual(sys.getrefcount(source), references)
                    np.testing.assert_array_equal(np.asarray(result), source)
