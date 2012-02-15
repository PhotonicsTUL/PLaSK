#include "geometry.h"

#include <plask/geometry/transform.h>

namespace plask { namespace python {

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

DECLARE_GEOMETRY_ELEMENT_23D(Translation, "Translation", "Holds a translated geometry element together with translation vector ("," version)")
{
    GEOMETRY_ELEMENT_23D(Translation, GeometryElementTransform<dim>, py::no_init)
    .def("__init__", py::make_constructor(&Translation_constructor1<dim>, py::default_call_policies(), (py::arg("child"), py::arg("translation"))))
    .def("__init__", py::make_constructor(&Translation_constructor2<dim>::call, py::default_call_policies(), Translation_constructor2<dim>::args))
    .def_readwrite("translation", &Translation<dim>::translation, "Translation vector")
    ;
}

void register_geometry_transform()
{
    init_Translation<2>();
    init_Translation<3>();
}

}} // namespace plask::python
