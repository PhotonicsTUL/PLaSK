import unittest
from numpy.testing import assert_allclose

import plask

from plask.util import gradients


plot_exact = True


class TestGradientXpl(unittest.TestCase):

    def setUp(self):
        manager = plask.Manager()
        manager.load('''
        <plask>
            <geometry>
                <cartesian2d name="original2d" axes="x,y">
                <stack>
                    <rectangle material="GaAs" dx="5." dy="1.0"/>
                    <rectangle material-top="Al(0.0)GaAs" material-bottom="Al(1.0)GaAs" dx="5." dy="0.2"/>
                    <rectangle material="AlAs" dx="5." dy="1.0"/>
                    <rectangle material-top="Al(1.0)GaAs" material-bottom="Al(0.0)GaAs" dx="5." dy="0.5"/>
                    <rectangle material="GaAs" dx="5." dy="1.0"/>
                </stack>
                </cartesian2d>
                <cartesian2d name="simplified2d" axes="x,y">
                <copy from="original2d">
                    <simplify-gradients lam="980"/>
                </copy>
                </cartesian2d>
            </geometry>
        </plask>
        ''')
        self.original2d = manager.geo.original2d
        self.simplified2d = manager.geo.simplified2d

    def test1(self):
        mesh = plask.mesh.Rectangular2D([2.], [1.01, 1.24, 1.26, 1.49])
        self.assertAlmostEqual(self.simplified2d.item[1][0][0][0].height, 0.5 * 0.49138, 5)
        self.assertAlmostEqual(self.simplified2d.item[1][0][0][0].height + self.simplified2d.item[1][0][1][0].height, 0.5, 5)
        assert_allclose(array(plask.MaterialField(self.simplified2d, mesh).nr(980.)),
                        array([3.4029+1.5121e-6j, 3.4029+1.5121e-6j, 3.0334+1.6038e-6j, 3.0334+1.6038e-6j]),
                        atol=1e-4)

    def test2(self):
        mesh = plask.mesh.Rectangular2D([2.], [2.51, 2.59, 2.61, 2.69])
        self.assertAlmostEqual(self.simplified2d.item[3][0][0][0].height, 0.2 * 0.50862, 5)
        self.assertAlmostEqual(self.simplified2d.item[3][0][0][0].height + self.simplified2d.item[3][0][1][0].height, 0.2, 5)
        assert_allclose(array(plask.MaterialField(self.simplified2d, mesh).nr(980.)),
                        array([3.0334+1.6038e-6j, 3.0334+1.6038e-6j, 3.4029+1.5121e-6j, 3.4029+1.5121e-6j]),
                        atol=1e-4)

    if __name__ == '__main__':
        def testPlot(self):
            mesh = plask.mesh.Rectangular2D([2.], plask.mesh.Regular(0.0, 3.7, 3701))
            global plot_exact
            if plot_exact:
                plask.plot_profile(plask.MaterialField(self.original2d, mesh).Nr(980.), color='#1f77b4', label='exact')
                plot_exact = False
            plask.plot_profile(plask.MaterialField(self.simplified2d, mesh).Nr(980.), color='#e5ae38', label='nr')


class TestGradientPython(unittest.TestCase):

    def setUp(self):
        self.original2d = plask.geometry.Stack2D()
        self.original2d.prepend(geometry.Rectangle(5., 1.0, 'GaAs'))
        self.original2d.prepend(geometry.Rectangle(5., 0.2, ('Al(1.0)GaAs', 'Al(0.0)GaAs')))
        self.original2d.prepend(geometry.Rectangle(5., 1.0, 'AlAs'))
        self.original2d.prepend(geometry.Rectangle(5., 0.5, ('Al(0.0)GaAs', 'Al(1.0)GaAs')))
        self.original2d.prepend(geometry.Rectangle(5., 1.0, 'GaAs'))

        self.simplified2d = gradients.simplify_all(self.original2d, 980., linear='eps')

    def test1(self):
        mesh = plask.mesh.Rectangular2D([2.], [1.01, 1.24, 1.26, 1.49])
        self.assertAlmostEqual(self.simplified2d[1][0][0][0].height, 0.5 * 0.500, 5)
        self.assertAlmostEqual(self.simplified2d[1][0][0][0].height + self.simplified2d[1][0][1][0].height, 0.5, 5)
        assert_allclose(array(plask.MaterialField(self.simplified2d, mesh).nr(980.)),
                        array([3.4073-3.6809e-7j, 3.4073-3.6809e-7j, 3.0385-3.6479e-7j, 3.0385-3.6479e-7j]),
                        atol=1e-4)

    def test2(self):
        mesh = plask.mesh.Rectangular2D([2.], [2.51, 2.59, 2.61, 2.69])
        self.assertAlmostEqual(self.simplified2d[3][0][0][0].height, 0.2 * 0.5000, 5)
        self.assertAlmostEqual(self.simplified2d[3][0][0][0].height + self.simplified2d[3][0][1][0].height, 0.2, 5)
        assert_allclose(array(plask.MaterialField(self.simplified2d, mesh).nr(980.)),
                        array([3.0385-3.6479e-7j, 3.0385-3.6479e-7j, 3.4073-3.6809e-7j, 3.4073-3.6809e-7j]),
                        atol=1e-4)

    if __name__ == '__main__':
        def testPlot(self):
            mesh = plask.mesh.Rectangular2D([2.], plask.mesh.Regular(0.0, 3.7, 3701))
            global plot_exact
            if plot_exact:
                plask.plot_profile(plask.MaterialField(self.original2d, mesh).Nr(980.), color='#1f77b4', label='exact')
                plot_exact = False
            plask.plot_profile(plask.MaterialField(self.simplified2d, mesh).Nr(980.), color='#d62728', label='exp')


if __name__ == '__main__':
    test = unittest.main(exit=False)
    plask.legend(loc='lower right')
    plask.show()
    sys.exit(not test.result.wasSuccessful())
