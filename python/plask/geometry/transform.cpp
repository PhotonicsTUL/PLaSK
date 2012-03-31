#include "geometry.h"

#include <plask/geometry/transform.h>

namespace plask { namespace python {

/// Initialize class GeometryElementTransform for Python
DECLARE_GEOMETRY_ELEMENT_23D(GeometryElementTransform, "GeometryElementTransform", "Base class for all "," transform nodes") {
    ABSTRACT_GEOMETRY_ELEMENT_23D(GeometryElementTransform, GeometryElementD<dim>)
        .add_property("child",
                      (shared_ptr<typename GeometryElementTransform<dim>::ChildType> (GeometryElementTransform<dim>::*)()) &GeometryElementTransform<dim>::getChild,
                      &GeometryElementTransform<dim>::setChild, "Child of the transform object")
        .def("hasChild", &GeometryElementTransform<dim>::hasChild, "Return true if the transform object has a set child")
    ;
}


// Some helpful wrappers
template <int dim>
shared_ptr<Translation<dim>> Translation_constructor1(shared_ptr<GeometryElementD<dim>> element, const Vec<dim,double>& trans) {
    return shared_ptr<Translation<dim>>(new Translation<dim>(element, trans));
}

template <int dim> struct Translation_constructor2 {};
template <> struct Translation_constructor2<2> {
    static inline shared_ptr<Translation<2>> call(shared_ptr<GeometryElementD<2>> element, double c0, double c1) {
        return shared_ptr<Translation<2>>(new Translation<2>(element, Vec<2,double>(c0, c1)));
    }
    const static py::detail::keywords<3> args;
};
const py::detail::keywords<3> Translation_constructor2<2>::args = (py::arg("child"), py::arg("c0"), py::arg("c1"));
template <> struct Translation_constructor2<3> {
    static inline shared_ptr<Translation<3>> call(shared_ptr<GeometryElementD<3>> element, double c0, double c1, double c2) {
        return shared_ptr<Translation<3>>(new Translation<3>(element, Vec<3,double>(c0, c1, c2)));
    }
    const static py::detail::keywords<4> args;
};
const py::detail::keywords<4> Translation_constructor2<3>::args = (py::arg("child"), py::arg("c0"), py::arg("c1"), py::arg("c2"));


DECLARE_GEOMETRY_ELEMENT_23D(Translation, "Translation", "Transfer holds a translated geometry element together with translation vector ("," version)")
{
    GEOMETRY_ELEMENT_23D(Translation, GeometryElementTransform<dim>, py::no_init)
    .def("__init__", py::make_constructor(&Translation_constructor1<dim>, py::default_call_policies(), (py::arg("child"), py::arg("translation"))))
    .def("__init__", py::make_constructor(&Translation_constructor2<dim>::call, py::default_call_policies(), Translation_constructor2<dim>::args))
    .def_readwrite("translation", &Translation<dim>::translation, "Translation vector")
    ;
}

void register_geometry_transform_background();

void register_geometry_transform()
{
    init_GeometryElementTransform<2>();
    init_GeometryElementTransform<3>();

    // Space changer
    py::class_<GeometryElementChangeSpace<3,2>, shared_ptr<GeometryElementChangeSpace<3,2>>, py::bases<GeometryElementTransform<3>>, boost::noncopyable>
    ("GeometryElementChangeSpace2Dto3D", "Base class for elements changing space 2D to 3D", py::no_init);

    py::class_<GeometryElementChangeSpace<2,3>, shared_ptr<GeometryElementChangeSpace<2,3>>, py::bases<GeometryElementTransform<2>>, boost::noncopyable>
    ("GeometryElementChangeSpace3Dto2D", "Base class for elements changing space 3D to 2D using some averaging or cross-section", py::no_init);

    init_Translation<2>();
    init_Translation<3>();

    register_geometry_transform_background();
}

}} // namespace plask::python
