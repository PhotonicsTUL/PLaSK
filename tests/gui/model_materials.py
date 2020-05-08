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

    def test_get_xml_element(self):
        self.assertEqualXML(self.material.get_xml_element(),
                            "<material name='custom_material' base='Al'><dens>1</dens><mobe>2*T</mobe></material>")

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
        self.materials.entries.append(MaterialsModel.Material(
            self.materials, "custom_material", "Al(0.5)GaN",
            [MaterialsModel.Material.Property('dens', '1', ['dans'])], endcomments=["propend"]))
        self.materials.entries.append(MaterialsModel.Material(
            self.materials, "second_material", "Ag", comments=["comment"]))
        self.materials.endcomments = ["end"]

    def test_get_xml_element(self):
        self.assertEqualXML(self.materials.get_xml_element(),
           "<materials>"
               "<material name='custom_material' base='Al(0.5)GaN'>"
                   "<!--dans-->"
                   "<dens>1</dens>"
                   "<!--propend-->"
               "</material>"
               "<!--comment-->"
               "<material name='second_material' base='Ag'/>"
               "<!--end-->"
           "</materials>")

    def test_set_xml_element(self):
        self.materials.set_xml_element(etree.XML(
            '''<materials>
                <material name="a" base="AuGe"/>
                <material name="b" base="Ag" alloy="yes"><!--mod--><mobe>3</mobe><!--bcom--></material>
                <!-- c comment -->
                <material name="c" base="In" alloy="yes"/>
                <!-- end comment -->
               </materials>'''))
        self.assertEqual(len(self.materials.entries), 3)
        self.assertEqual(self.materials.entries[0].name, 'a')
        self.assertEqual(self.materials.entries[0].base, 'AuGe')
        self.assertEqual(self.materials.entries[0].alloy, False)
        self.assertEqual(len(self.materials.entries[0].properties), 0)
        self.assertEqual(self.materials.entries[1].name, 'b')
        self.assertEqual(self.materials.entries[1].base, 'Ag')
        self.assertEqual(self.materials.entries[1].alloy, True)
        self.assertEqual(len(self.materials.entries[1].properties), 1)
        self.assertEqual(self.materials.entries[1].properties[0].name, 'mobe')
        self.assertEqual(self.materials.entries[1].properties[0].value, '3')
        self.assertEqual(self.materials.entries[1].properties[0].comments, ['mod'])
        self.assertEqual(self.materials.entries[1].endcomments, ['bcom'])
        self.assertEqual(self.materials.entries[2].name, 'c')
        self.assertEqual(self.materials.entries[2].base, 'In')
        self.assertEqual(self.materials.entries[2].alloy, True)
        self.assertEqual(self.materials.entries[2].comments, [" c comment "])
        self.assertEqual(len(self.materials.entries[2].properties), 0)
        self.assertEqual(self.materials.endcomments, [" end comment "])

    def test_get(self):
        self.assertEqual(self.materials.get(0, 0), "custom_material")
        self.assertEqual(self.materials.get(1, 0), "Al(0.5)GaN")
        self.assertEqual(self.materials.get(2, 0), False)
        self.assertEqual(self.materials.get(0, 1), "second_material")
        self.assertEqual(self.materials.get(1, 1), "Ag")
        self.assertEqual(self.materials.get(2, 1), False)

    def test_set(self):
        self.materials.set(0, 0, "new_name")
        self.materials.set(2, 0, True)
        self.materials.set(1, 1, "AuGe")
        self.assertEqual(self.materials.get(0, 0), "new_name")
        self.assertEqual(self.materials.get(1, 0), "Al(0.5)GaN")
        self.assertEqual(self.materials.get(2, 0), True)
        self.assertEqual(self.materials.get(0, 1), "second_material")
        self.assertEqual(self.materials.get(1, 1), "AuGe")
        self.assertEqual(self.materials.get(2, 1), False)

    def test_columnCount(self):
        self.assertEqual(self.materials.columnCount(), 3)

    def test_create_info_with_no_name(self):
        self.materials.entries[1].name = ''   # no required name
        info = self.materials.create_info()
        self.assertEqual(len(info), 1)
        self.assertCountEqual(info[0].rows, (1,))
        self.assertCountEqual(info[0].cols, (0,))

    def test_create_info_with_no_name(self):
        self.materials.entries[0].base = ''   # no required base
        info = self.materials.create_info()
        self.assertEqual(len(info), 1)
        self.assertCountEqual(info[0].rows, (0,))
        self.assertCountEqual(info[0].cols, (1,))

    def test_create_info_no_dopant(self):
        self.materials.entries[0].base = 'AlGaN'   # no required dopant
        info = self.materials.create_info()
        self.assertEqual(len(info), 1)
        self.assertCountEqual(info[0].rows, (0,))
        self.assertCountEqual(info[0].cols, (1,))



if __name__ == '__main__':
    test = unittest.main(exit=False)
    sys.exit(not test.result.wasSuccessful())


