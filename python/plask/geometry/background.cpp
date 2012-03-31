#include "geometry.h"

#include <plask/geometry/background.h>

namespace plask { namespace python {


template<int dim>
static typename Background<dim>::ExtendType _Background_makeExtend(const std::string& extend_str) {}

template<>
typename Background<2>::ExtendType _Background_makeExtend<2>(const std::string& extend_str) {
    if (extend_str == "all") return Background<2>::EXTEND_ALL;
    int extend = Background<2>::EXTEND_NONE;
    if (Config::z_up){
        for (auto c: extend_str) {
            c = std::tolower(c);
            if (c == 'y' || c == 'r' || c == '0') extend |= Background<2>::EXTEND_TRAN;
            else if (c == 'z' || c == '1') extend |= Background<2>::EXTEND_VERTICAL;
            else {
                PyErr_SetString(PyExc_ValueError, "wrong axis in 'extend' for config.vertical_axis='z'");
                throw py::error_already_set();
            }
        }
    } else {
        for (auto c: extend_str) {
            c = std::tolower(c);
            if (c == 'x' || c == '0') extend |= Background<2>::EXTEND_TRAN;
            else if (c == 'y' || c == '1') extend |= Background<2>::EXTEND_VERTICAL;
            else {
                PyErr_SetString(PyExc_ValueError, "wrong axis in 'extend' for config.vertical_axis='y'");
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
            c = std::tolower(c);
            if (c == 'x' || c == 'r' || c == '0') extend |= Background<3>::EXTEND_LON;
            else if (c == 'y' || c == 'p' || c == '1') extend |= Background<3>::EXTEND_TRAN;
            else if (c == 'z' || c == '2') extend |= Background<3>::EXTEND_VERTICAL;
            else {
                PyErr_SetString(PyExc_ValueError, "wrong axis in 'extend' for config.vertical_axis='z'");
                throw py::error_already_set();
            }
        }
    } else {
        for (auto c: extend_str) {
            c = std::tolower(c);
            if (c == 'z' || c == '0') extend |= Background<3>::EXTEND_TRAN;
            else if (c == 'x' || c == '1') extend |= Background<3>::EXTEND_TRAN;
            else if (c == 'y' || c == '2') extend |= Background<3>::EXTEND_VERTICAL;
            else {
                PyErr_SetString(PyExc_ValueError, "wrong axis in 'extend' for config.vertical_axis='y'");
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
static std::string Background_getExtend(const Background<dim>& self) {}

template<>
std::string Background_getExtend(const Background<2>& self) {
    std::string result;
    if (self.getExtend() & Background<2>::EXTEND_TRAN) result += "0";
    if (self.getExtend() & Background<2>::EXTEND_VERTICAL) result += "1";
    return result;
}

template<>
std::string Background_getExtend(const Background<3>& self) {
    std::string result;
    if (self.getExtend() & Background<3>::EXTEND_LON) result += "0";
    if (self.getExtend() & Background<3>::EXTEND_TRAN) result += "1";
    if (self.getExtend() & Background<3>::EXTEND_VERTICAL) result += "2";
    return result;
}



DECLARE_GEOMETRY_ELEMENT_23D(Background, "Background",
    "This is a transparent container for a single element. However all for material queries\n"
    "it considers the points outside of its bounding box as if they were ocated exactly at\n"
    "the edges of the bounding box. This allows to create infinite egde areas filled with\n"
    "some particular material.\n\n"
    "This container is meant to be used as the root of the geometry tree.\n\n"
    "Background","(child=None, along=None)\n"
    "    Create background object and add a child to it. 'along' denotes axes\n"
    "    along which the materials are extended. It can be a string combining\n"
    "    axes symbols or 'all' for every direction.")
{
    GEOMETRY_ELEMENT_23D(Background, GeometryElementTransform<dim>, py::no_init)
        .def("__init__", py::make_constructor(&Background__init__<dim>, py::default_call_policies(),
                                              (py::arg("child")=shared_ptr<GeometryElementD<dim>>(), py::arg("along")="all")))
        .add_property("extend", &Background_getExtend<dim>, &Background_setExtend<dim>, "Directions of extension")
    ;
}


void register_geometry_transform_background()
{
    init_Background<2>();
    init_Background<3>();
}

}} // namespace plask::python
