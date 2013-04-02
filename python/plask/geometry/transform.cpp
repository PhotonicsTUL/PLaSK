#include "geometry.h"

#include <plask/geometry/transform.h>
#include <plask/geometry/mirror.h>

namespace plask { namespace python {

template <int dim>
static bool Transfrom__contains__(const GeometryObjectTransform<dim>& self, shared_ptr<typename GeometryObjectTransform<dim>::ChildType> child) {
    if (self.getChild() == child) return true;
    return false;
}


/// Initialize class GeometryObjectTransform for Python
DECLARE_GEOMETRY_ELEMENT_23D(GeometryObjectTransform, "GeometryObjectTransform", "Base class for all "," transform nodes") {
    ABSTRACT_GEOMETRY_ELEMENT_23D(GeometryObjectTransform, GeometryObjectD<dim>)
        .add_property("item",
                      (shared_ptr<typename GeometryObjectTransform<dim>::ChildType> (GeometryObjectTransform<dim>::*)()) &GeometryObjectTransform<dim>::getChild,
                      &GeometryObjectTransform<dim>::setChild, "Child of the transform object")
        .def("__contains__", &Transfrom__contains__<dim>)
    ;
}


// Some helpful wrappers
template <int dim>
shared_ptr<Translation<dim>> Translation_constructor1(shared_ptr<GeometryObjectD<dim>> object, const Vec<dim,double>& trans) {
    return make_shared<Translation<dim>>(object, trans);
}

template <int dim> struct Translation_constructor2 {};
template <> struct Translation_constructor2<2> {
    static inline shared_ptr<Translation<2>> call(shared_ptr<GeometryObjectD<2>> object, double c0, double c1) {
        return make_shared<Translation<2>>(object, Vec<2,double>(c0, c1));
    }
    const static py::detail::keywords<3> args;
};
const py::detail::keywords<3> Translation_constructor2<2>::args = (py::arg("item"), py::arg("c0"), py::arg("c1"));
template <> struct Translation_constructor2<3> {
    static inline shared_ptr<Translation<3>> call(shared_ptr<GeometryObjectD<3>> object, double c0, double c1, double c2) {
        return make_shared<Translation<3>>(object, Vec<3,double>(c0, c1, c2));
    }
    const static py::detail::keywords<4> args;
};
const py::detail::keywords<4> Translation_constructor2<3>::args = (py::arg("item"), py::arg("c0"), py::arg("c1"), py::arg("c2"));


std::string GeometryObject__repr__(const shared_ptr<GeometryObject>& self);

template <int dim>
static std::string Translation__str__(const Translation<dim>& self) {
    std::stringstream out;
    out << "(";
    try {
        std::string str = py::extract<std::string>(py::object(self.getChild()).attr("__repr__")());
        out << str;
    } catch (py::error_already_set) {
        PyErr_Clear();
        out << GeometryObject__repr__(self.getChild());
    }
    out << ", plask.vec("; for (int i = 0; i < dim; ++i) out << pyformat(self.translation[i]) << (i!=dim-1 ? "," : ")");
    out << ")";
    return out.str();
}

template <int dim>
static std::string Translation__repr__(const Translation<dim>& self) {
    return format("plask.geometry.Translation%1%D%2%", dim, Translation__str__<dim>(self));
}

DECLARE_GEOMETRY_ELEMENT_23D(Translation, "Translation", "Transform that holds a translated geometry object together with translation vector ("," version)")
{
    GEOMETRY_ELEMENT_23D(Translation, GeometryObjectTransform<dim>, py::no_init)
    .def("__init__", py::make_constructor(&Translation_constructor1<dim>, py::default_call_policies(), (py::arg("item"), py::arg("translation"))))
    .def("__init__", py::make_constructor(&Translation_constructor2<dim>::call, py::default_call_policies(), Translation_constructor2<dim>::args))
    .def_readwrite("translation", &Translation<dim>::translation, "Translation vector")
    .def("__str__", &Translation__str__<dim>)
    .def("__repr__", &Translation__repr__<dim>)
    ;
}


template <typename Cls>
shared_ptr<Cls> Mirror_constructor1(size_t axis, shared_ptr<typename Cls::ChildType> child) {
    if (axis >= Cls::DIM) throw ValueError("Wrong axis number.");
    return make_shared<Cls>(typename Primitive<Cls::DIM>::Direction(axis), child);
}

template <typename Cls>
shared_ptr<Cls> Mirror_constructor2(const std::string& axis, shared_ptr<typename Cls::ChildType> child) {
    size_t no = config.axes[axis] + Cls::DIM - 3;
    return make_shared<Cls>(typename Primitive<Cls::DIM>::Direction(no), child);
}

template <typename Cls>
std::string getFlipDir(const Cls& self) { return config.axes[self.flipDir]; }

template <typename Cls>
void setFlipDir(Cls& self, py::object val) {
    try {
        size_t no = config.axes[py::extract<std::string>(val)] + Cls::DIM - 3;
        self.flipDir = typename Primitive<Cls::DIM>::Direction(no);
    } catch (py::error_already_set) {
        PyErr_Clear();
        size_t no = py::extract<size_t>(val);
        if (no >= Cls::DIM) throw ValueError("Wrong axis number.");
        self.flipDir = typename Primitive<Cls::DIM>::Direction(no);
    }
}

DECLARE_GEOMETRY_ELEMENT_23D(Flip, "Flip", "Transfer that flips the geometry object along axis specified by name or number ("," version)")
{
    GEOMETRY_ELEMENT_23D(Flip, GeometryObjectTransform<dim>, py::no_init)
    .def("__init__", py::make_constructor(&Mirror_constructor1<Flip<dim>>, py::default_call_policies(), (py::arg("axis"), py::arg("item")=shared_ptr<GeometryObjectD<dim>>())))
    .def("__init__", py::make_constructor(&Mirror_constructor2<Flip<dim>>, py::default_call_policies(), (py::arg("axis"), py::arg("item")=shared_ptr<GeometryObjectD<dim>>())))
    .add_property("axis", &getFlipDir<Flip<dim>>, &setFlipDir<Flip<dim>>, "Flip axis")
    ;
}

DECLARE_GEOMETRY_ELEMENT_23D(Mirror, "Mirror", "Transfer that mirrors the geometry object along axis specified by name or number ("," version)")
{
    GEOMETRY_ELEMENT_23D(Mirror, GeometryObjectTransform<dim>, py::no_init)
    .def("__init__", py::make_constructor(&Mirror_constructor1<Mirror<dim>>, py::default_call_policies(), (py::arg("axis"), py::arg("item")=shared_ptr<GeometryObjectD<dim>>())))
    .def("__init__", py::make_constructor(&Mirror_constructor2<Mirror<dim>>, py::default_call_policies(), (py::arg("axis"), py::arg("item")=shared_ptr<GeometryObjectD<dim>>())))
    .add_property("axis", &getFlipDir<Mirror<dim>>, &setFlipDir<Mirror<dim>>, "Mirror axis")
    ;
}

void register_geometry_changespace();

void register_geometry_transform()
{
    init_GeometryObjectTransform<2>();
    init_GeometryObjectTransform<3>();

    register_geometry_changespace();

    init_Translation<2>();
    init_Translation<3>();

    init_Flip<2>();
    init_Flip<3>();

    init_Mirror<2>();
    init_Mirror<3>();
}

}} // namespace plask::python
