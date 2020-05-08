import sys, os
import unittest
sys.path.insert(0, os.path.abspath('../..'))

from gui_test_utils import GUITestCase

from gui.utils.validators import can_be_float, can_be_int, can_be_bool


class TestGUIUtilsValidators(GUITestCase):

    def test_can_be_float(self):
        self.assertTrue(can_be_float('1.5'))
        self.assertTrue(can_be_float('1.5', required=True))
        self.assertTrue(can_be_float('1.5', float_validator=lambda x: x == 1.5))
        self.assertFalse(can_be_float('1.5', float_validator=lambda x: x == 2.5))
        self.assertTrue(can_be_float('-1.5e-10'))
        self.assertFalse(can_be_float('float'))
        self.assertFalse(can_be_float('float', float_validator=lambda x: True))
        self.assertTrue(can_be_float('{a+1.5}'))
        self.assertTrue(can_be_float('{a*b}'))
        self.assertFalse(can_be_float('a{a*b}'))
        self.assertFalse(can_be_float('{a*b}a'))
        self.assertTrue(can_be_float(''))
        self.assertTrue(can_be_float(' '))
        self.assertTrue(can_be_float(None))
        self.assertFalse(can_be_float('', required=True))
        self.assertFalse(can_be_float(' ', required=True))
        self.assertFalse(can_be_float(None, required=True))

    def test_can_be_int(self):
        self.assertTrue(can_be_int('2'))
        self.assertTrue(can_be_int('2', required=True))
        self.assertFalse(can_be_int('2.5'))
        self.assertFalse(can_be_int('int'))
        self.assertFalse(can_be_int('int', int_validator=lambda x: True))
        self.assertTrue(can_be_int('{a+1}'))
        self.assertTrue(can_be_int('{a*b}'))
        self.assertFalse(can_be_int('a{a*b}'))
        self.assertFalse(can_be_int('{a*b}a'))
        self.assertTrue(can_be_int(''))
        self.assertTrue(can_be_int(' '))
        self.assertTrue(can_be_int(None))
        self.assertFalse(can_be_int('', required=True))
        self.assertFalse(can_be_int(' ', required=True))
        self.assertFalse(can_be_int(None, required=True))

    def test_can_be_bool(self):
        self.assertTrue(can_be_bool('True'))
        self.assertTrue(can_be_bool('false', required=True))
        self.assertTrue(can_be_bool('yes'))
        self.assertTrue(can_be_bool('NO'))
        self.assertTrue(can_be_bool('0'))
        self.assertTrue(can_be_bool('1'))
        self.assertFalse(can_be_bool('2'))
        self.assertFalse(can_be_bool('bool'))
        self.assertTrue(can_be_bool('{not a}'))
        self.assertTrue(can_be_bool(''))
        self.assertTrue(can_be_bool(' '))
        self.assertTrue(can_be_bool(None))
        self.assertFalse(can_be_bool('', required=True))
        self.assertFalse(can_be_bool(' ', required=True))
        self.assertFalse(can_be_bool(None, required=True))


if __name__ == '__main__':
    test = unittest.main(exit=False)
    sys.exit(not test.result.wasSuccessful())