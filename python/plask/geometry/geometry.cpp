#include <cmath>
#include "../python_globals.h"
#include <boost/python/stl_iterator.hpp>
#include "../../util/raw_constructor.h"

#include <plask/config.h>
#include <plask/exceptions.h>
#include <plask/geometry/leaf.h>
#include <plask/utils/format.h>

namespace plask { namespace python {

namespace py = boost::python;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void register_calculation_spaces();

void register_geometry_aligners();
void register_geometry_object();
void register_geometry_primitive();
void register_geometry_leafs();
void register_geometry_transform();
void register_geometry_aligners();
void register_geometry_path();
void register_geometry_container();

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void initGeometry() {

    py::object geometry_module { py::handle<>(py::borrowed(PyImport_AddModule("plask.geometry"))) };
    py::scope().attr("geometry") = geometry_module;
    py::scope scope = geometry_module;

    scope.attr("__doc__") =
        "This solver provides 2D and 3D geometry objects, necessary to describe the structure "
        "of analyzed device."; //TODO maybe more extensive description

    // This must be the first one
    register_geometry_path();


    register_geometry_object();
    register_geometry_primitive();
    register_geometry_leafs();
    register_geometry_transform();
    register_geometry_aligners();
    register_geometry_container();


    register_exception<NoSuchGeometryObject>(PyExc_IndexError);


    // Calculation spaces
    register_calculation_spaces();
}

}} // namespace plask::python
