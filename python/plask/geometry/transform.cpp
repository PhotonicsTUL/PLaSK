#include "geometry.h"

#include <plask/geometry/transform.h>
#include <plask/geometry/background.h>

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

template<int dim>
static typename Background<dim>::ExtendType _Background_makeExtend(const std::string& extend_str) {}

template<>
typename Background<2>::ExtendType _Background_makeExtend<2>(const std::string& extend_str) {
    if (extend_str == "all") return Background<2>::EXTEND_ALL;
    int extend = Background<2>::EXTEND_NONE;
    if (Config::z_up){
        for (auto c: extend_str) {
            if (c == 'y' || c == 'r' || c == '0') extend |= Background<2>::EXTEND_TRAN;
            else if (c == 'z' || c == '1') extend |= Background<2>::EXTEND_VERTICAL;
            else {
                PyErr_SetString(PyExc_ValueError, "wrong axis for 'extend' for config.vertical_axis='z'");
                throw py::error_already_set();
            }
        }
    } else {
        for (auto c: extend_str) {
            if (c == 'x' || c == '0') extend |= Background<2>::EXTEND_TRAN;
            else if (c == 'y' || c == '1') extend |= Background<2>::EXTEND_VERTICAL;
            else {
                PyErr_SetString(PyExc_ValueError, "wrong axis for 'extend' for config.vertical_axis='y'");
                throw py::error_already_set();
            }
        }
    }
    return Background<2>::ExtendType(extend);
}

template<>
typename Background<3>::ExtendType _Background_makeExtend<3>(const std::string& extend_str) {
    if (extend_str == "all") return Background<3>::EXTEND_ALL;
    int extend = Background<3>::EXTEND_NONE;
    if (Config::z_up){
        for (auto c: extend_str) {
            if (c == 'x' || c == 'r' || c == '0') extend |= Background<3>::EXTEND_LON;
            else if (c == 'y' || c == 'p' || c == '1') extend |= Background<3>::EXTEND_TRAN;
            else if (c == 'z' || c == '2') extend |= Background<3>::EXTEND_VERTICAL;
            else {
                PyErr_SetString(PyExc_ValueError, "wrong axis for 'extend' for config.vertical_axis='z'");
                throw py::error_already_set();
            }
        }
    } else {
        for (auto c: extend_str) {
            if (c == 'z' || c == '0') extend |= Background<3>::EXTEND_TRAN;
            else if (c == 'x' || c == '1') extend |= Background<3>::EXTEND_TRAN;
            else if (c == 'y' || c == '2') extend |= Background<3>::EXTEND_VERTICAL;
            else {
                PyErr_SetString(PyExc_ValueError, "wrong axis for 'extend' for config.vertical_axis='y'");
                throw py::error_already_set();
            }
        }
    }
    return Background<3>::ExtendType(extend);
}

template<int dim>
static shared_ptr<Background<dim>> Background__init__(shared_ptr<GeometryElementD<dim>> child, const std::string& extend) {
    return make_shared<Background<dim>>(child, _Background_makeExtend<dim>(extend));
}

template<int dim>
static void Background_setExtend(Background<dim>& self, const std::string& extend) {
    self.setExtend(_Background_makeExtend<dim>(extend));
}

template<int dim>
static std::string Background_getExtend(const Background<dim>& self) { return "TODO"; } //TODO


DECLARE_GEOMETRY_ELEMENT_23D(Background, "Background",
    "This is a transparent container for a single element. However all for material queries\n"
    "it considers the points outside of its bounding box as if they were ocated exactly at\n"
    "the edges of the bounding box. This allows to create infinite egde areas filled with\n"
    "some particular material.\n\n"
    "This container is meant to be used as the root of the geometry tree.\n\n"
    "Background","(child=None, axes=None)\n"
    "    Create background object and add a child to it. Axes denote directions in which\n"
    "    the materials are extended. It can be a string combining axes symbols or 'all'.")
{
    GEOMETRY_ELEMENT_23D(Background, GeometryElementTransform<dim>, py::no_init)
        .def("__init__", py::make_constructor(&Background__init__<dim>, py::default_call_policies(),
                                              (py::arg("child")=shared_ptr<GeometryElementD<dim>>(), py::arg("extend")="all")))
        .add_property("extend", &Background_getExtend<dim>, &Background_setExtend<dim>, "Directions of extension")
    ;
}


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

    init_Background<2>();
    init_Background<3>();
}

}} // namespace plask::python
