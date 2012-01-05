#include <boost/python.hpp>
namespace py = boost::python;

#include <config.h>
#include <plask/geometry/manager.h>
#include <plask/geometry/leaf.h>

#include "geometry/element.h"
#include "geometry/primitive.h"
#include "geometry/leafs.h"
#include "geometry/container.h"

namespace plask { namespace python {


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void initGeometry() {

    py::object geometry_module { py::handle<>(py::borrowed(PyImport_AddModule("plask.geometry"))) };
    py::scope().attr("geometry") = geometry_module;
    py::scope scope = geometry_module;

    scope.attr("__doc__") =
        "This module provides 2D and 3D geometry elements, necessary to describe the structure "
        "of analyzed device."; //TODO maybe more extensive description


    register_geometry_element_h();
    register_geometry_primitive_h();
    register_geometry_leafs();
    register_geometry_container_h();

    // manager.h

//     py::class_<GeometryManager>("Geometry",
//         "Main geometry manager. It manages the whole geometry of analyzed device "
//         "and provides methods to read and write it to an XML file.")
//
//     ;


}

}} // namespace plask::python