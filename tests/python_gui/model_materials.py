import sys, os
import unittest
sys.path.insert(0, os.path.abspath('../..'))

from gui_test_utils import GUITestCase
from gui.model.materials import MaterialsModel, parse_material_components, material_unit

from lxml import etree

class TestGUIModelMaterial(GUITestCase):

    def setUp(self):
        self.materials = MaterialsModel()
        self.material = MaterialsModel.Material(self.materials, "custom_material", "Al",
                                                [MaterialsModel.Material.Property('dens', '1'),
                                                MaterialsModel.Material.Property('mobe', '2*T')])
        self.materials.entries.append(self.material)

    def test_rowCount(self):
        self.assertEqual(self.material.rowCount(), 2)   # 2 properties

    def test_columnCount(self):
        self.assertEqual(self.material.columnCount(), 4)

    def test_get(self):
        self.assertEqual(self.material.get(0, 0), 'dens')
        self.assertEqual(self.material.get(1, 0), '1')
        self.assertEqual(self.material.get(2, 0), 'kg/m<sup>3</sup>')
        self.assertIsInstance(self.material.get(3, 0), str) # HTML help
        self.assertEqual(self.material.get(0, 1), 'mobe')
        self.assertEqual(self.material.get(1, 1), '2*T')
        self.assertEqual(self.material.get(2, 1), 'cm<sup>2</sup>/(V s)')
        self.assertIsInstance(self.material.get(3, 1), str) # HTML help

    def test_set(self):
        self.material.set(0, 1, 'Eg')
        self.material.set(1, 1, '2')
        self.assertEqual(self.material.properties[0].name, 'dens')
        self.assertEqual(self.material.properties[0].value, '1')
        self.assertEqual(self.material.properties[1].name, 'Eg')
        self.assertEqual(self.material.properties[1].value, '2')


class TestGUIModelMaterials(GUITestCase):

    def setUp(self):
        self.materials = MaterialsModel()
        self.materials.entries.append(MaterialsModel.Material(self.materials, "custom_material", "AlGaN"))


if __name__ == '__main__':
    test = unittest.main(exit=False)
    sys.exit(not test.result.wasSuccessful())


