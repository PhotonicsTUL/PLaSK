#include "geometry.h"

#include <plask/geometry/space_changer_cartesian.h>

namespace plask { namespace python {

template <int dim, int cdim>
static bool STransfrom__contains__(const GeometryElementChangeSpace<dim,cdim>& self, shared_ptr<typename GeometryElementTransform<cdim>::ChildType> child) {
    if (self.getChild() == child) return true;
    return false;
}


void register_geometry_changespace()
{
    py::class_<GeometryElementChangeSpace<3,2>, shared_ptr<GeometryElementChangeSpace<3,2>>, py::bases<GeometryElementD<3>>, boost::noncopyable>
    ("GeometryElementChangeSpace2Dto3D", "Base class for elements changing space 2D to 3D", py::no_init)
        .add_property("child",
                      (shared_ptr<typename GeometryElementChangeSpace<3,2>::ChildType> (GeometryElementChangeSpace<3,2>::*)()) &GeometryElementChangeSpace<3,2>::getChild,
                      &GeometryElementChangeSpace<3,2>::setChild, "Child of the transform object")
        .def("__contains__", &STransfrom__contains__<3,2>)
    ;

    // py::class_<GeometryElementChangeSpace<2,3>, shared_ptr<GeometryElementChangeSpace<2,3>>, py::bases<GeometryElementD<2>>, boost::noncopyable>
    // ("GeometryElementChangeSpace3Dto2D", "Base class for elements changing space 3D to 2D using some averaging or cross-section", py::no_init)
    //     .add_property("child",
    //                   (shared_ptr<typename GeometryElementChangeSpace<2,3>::ChildType> (GeometryElementChangeSpace<2,3>::*)()) &GeometryElementChangeSpace<2,3>::getChild,
    //                   &GeometryElementChangeSpace<2,3>::setChild, "Child of the transform object")
    //     .def("__contains__", &STransfrom__contains__<2,3>)
    // ;

    py::class_<Extrusion, shared_ptr<Extrusion>, py::bases<GeometryElementChangeSpace<3,2>>, boost::noncopyable>("Extrusion",
        "Extrusion in the longitudinal direction of a 2D object to the 3D one."
        "Objects of this type can be supplied to 2D Cartesian modules or they can be used\n"
        "as a part of the 3D geometry tree.\n\n"
        "Extrusion(child=None, length=infinity)\n    Create new extrusion of given length with provided child",
        py::init<shared_ptr<GeometryElementD<2>>,double>((py::arg("child")=shared_ptr<GeometryElementD<2>>(), py::arg("length")=INFINITY)))
        .def_readwrite("length", &Extrusion::length, "Length of the extrusion in the longitudinal direction")
    ;
}

}} // namespace plask::python
