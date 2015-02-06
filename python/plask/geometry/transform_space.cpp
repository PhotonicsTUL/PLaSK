#include "geometry.h"

#include <plask/geometry/transform_space_cartesian.h>
#include <plask/geometry/transform_space_cylindric.h>

namespace plask { namespace python {

template <int dim, int cdim>
static bool STransfrom__contains__(const GeometryObjectTransformSpace<dim,cdim>& self, shared_ptr<typename GeometryObjectTransform<cdim>::ChildType> child) {
    if (self.getChild() == child) return true;
    return false;
}


void register_geometry_changespace()
{
    py::class_<GeometryObjectTransformSpace<3,2>, shared_ptr<GeometryObjectTransformSpace<3,2>>, py::bases<GeometryObjectD<3>>, boost::noncopyable>
    ("GeometryObjectTransform2Dto3D", "Base class for all transformations which change 2D space to 3D.", py::no_init)
        .add_property("item",
                      (shared_ptr<typename GeometryObjectTransformSpace<3,2>::ChildType> (GeometryObjectTransformSpace<3,2>::*)()) &GeometryObjectTransformSpace<3,2>::getChild,
                      &GeometryObjectTransformSpace<3,2>::setChild, "Transformed 2D object.")
        .def("__contains__", &STransfrom__contains__<3,2>)
    ;

    // py::class_<GeometryObjectTransformSpace<2,3>, shared_ptr<GeometryObjectTransformSpace<2,3>>, py::bases<GeometryObjectD<2>>, boost::noncopyable>
    // ("GeometryObjectTransformGeometry3Dto2D", "Base class for objects changing space 3D to 2D using some averaging or cross-section", py::no_init)
    //     .add_property("item",
    //                   (shared_ptr<typename GeometryObjectTransformSpace<2,3>::ChildType> (GeometryObjectTransformSpace<2,3>::*)()) &GeometryObjectTransformSpace<2,3>::getChild,
    //                   &GeometryObjectTransformSpace<2,3>::setChild, "Child of the transform object")
    //     .def("__contains__", &STransfrom__contains__<2,3>)
    // ;

    py::class_<Extrusion, shared_ptr<Extrusion>, py::bases<GeometryObjectTransformSpace<3,2>>, boost::noncopyable>("Extrusion",
        "Extrusion(item, length=infinity)\n\n"
        "Extrusion in the longitudinal direction of the 2D object into a 3D one.\n\n"
        "Objects of this type can be supplied to 2D Cartesian solvers or they can be used\n"
        "as a part of the 3D geometry tree.\n\n"
        "Args:\n"
        "    item (2D geometry object): Two-dimensional geometry object to extrude.\n"
        "    length (float): Extrusion length in the longitudinal direction.\n\n"
        "Example:\n"
        "    Instead of using :class:`~plask.geometry.Cuboid` object, you can extrude\n"
        "    a rectangle in the following way:\n\n"
        "    >>> rect = plask.geometry.Rectangle(2, 3, 'GaAs')\n"
        "    >>> cuboid = plask.geometry.Extrusion(rect, 1)\n\n"
        "    The created cuboid will have dimensions 1µm, 2µm, and 3µm along logitudinal,\n"
        "    transverse, and vertical axes, respectively.\n\n"
        "    **Note:** In the real-life situations, you can extrude any complicated 2D\n"
        "    object (e.g. a stack of a shelf).\n",
        py::init<shared_ptr<GeometryObjectD<2>>,double>((py::arg("item"), py::arg("length")=INFINITY)))
        .add_property<>("length", &Extrusion::getLength, &Extrusion::setLength, "Length of the extrusion in the longitudinal direction.")
    ;

    py::class_<Revolution, shared_ptr<Revolution>, py::bases<GeometryObjectTransformSpace<3,2>>, boost::noncopyable>("Revolution",
        "Revolution(item)\n\n"
        "Revolution around the vertical axis of the 2D object into a 3D one.\n\n"
        "Objects of this type can be supplied to 2D cylindrical solvers or they can be\n"
        "used as a part of the 3D geometry tree.\n\n"
        "Args:\n"
        "    item (2D geometry object): Two-dimensional geometry object to revolve.\n"
        "    auto_clip (bool): If True, item will be implicitly clipped to non-negative tran. coordinates. False by default.\n\n"
        "Example:\n"
        "    Instead of using :class:`~plask.geometry.Cylinder` object, you can revolve\n"
        "    a rectangle in the following way:\n\n"
        "    >>> rect = plask.geometry.Rectangle(1, 2, 'GaAs')\n"
        "    >>> cylinder = plask.geometry.Revolution(rect)\n\n"
        "    The created cylinder will have base radius of 1µm and the height 2µm.\n\n"
        "    **Note:** In the real-life situations, you can revolve any complicated 2D\n"
        "    object (e.g. a stack of a shelf).\n",
        py::init<shared_ptr<GeometryObjectD<2>>>((py::arg("item"), py::arg("auto_clip")=false)))
    ;
}

}} // namespace plask::python
