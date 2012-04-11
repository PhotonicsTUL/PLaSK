#include "../globals.h"

#include <plask/geometry/calculation_space.h>

namespace plask { namespace python {


shared_ptr <Space2DCartesian> Space2DCartesian__init__(py::tuple args, py::dict kwargs) {
    int na = py::len(args);

    shared_ptr <Space2DCartesian> space;

    if (na == 3) {
        shared_ptr<GeometryElementD<2>> element = py::extract<shared_ptr<GeometryElementD<2>>>(args[1]);
        double length = py::extract<double>(args[2]);
        space = make_shared<Space2DCartesian>(element, length);
    } else {
        PyErr_SetString();
    }

    return make_shared<Space2DCartesian>(extrusion);
    return
}


void register_calculation_spaces() {

    py::class_<Space2DCartesian, shared_ptr<Space2DCartesian>>("Space2DCartesian",
        "Calculation space representing 2D Cartesian coordinate system\n\n"
        "Space2DCartesian(extrusion, **borders)\n"
        "    Create a space around the provided extrusion\n\n",
        "Space2DCartesian(geometry_element, length=infty, **borders)\n"
        "    Create a space around the two-dimensional geometry element with given length\n\n"
        "Borders " //TODO
        py::noinit)
        .def("__init__", raw_constructor(Space2DCartesian__init__, 1))
    ;


}


}} // namespace plask::python
