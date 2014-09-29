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
        "PLaSK geometry classes.\n\n"

        "Classes and functions defined in this module can be used to create and modify\n"
        "geometry description in PLaSK. See :ref:`sec-geometry-python` for more details.\n\n"

        "Example:\n"
        "    To create a simple stack with two identical rectangles and check its total\n"
        "    size, use the following commands:\n\n"

        "    >>> rectangle = geometry.Block2D(4, 2, 'GaAs')\n"
        "    >>> stack = geometry.Stack2D()\n"
        "    >>> stack.prepend(rectangle)\n"
        "    <plask.geometry.PathHint at 0x40a52f8>\n"
        "    >>> stack.prepend(rectangle)\n"
        "    <plask.geometry.PathHint at 0x40a50d8>\n"
        "    >>> stack.bbox\n"
        "    plask.geometry.Box2D(0, 0, 4, 4)\n\n"

        "    Now, to create a Cartesian two-dimensional geometry over it:\n\n"

        "    >>> geometry.Cartesian2D(stack)\n"
        "    <plask.geometry.Cartesian2D object at (0x571acd0)>\n\n"

        "    You may also modify any existing geometry object:\n\n"

        "    >>> rectangle.height = 3\n"
        "    >>> stack.bbox\n"
        "    plask.geometry.Box2D(0, 0, 4, 6)\n\n"

        "See also:\n"
        "    Section :ref:`sec-geometry` for detailed information of geometry.\n\n"

        "    XPL section :xml:tag:`geometry` for reference of the geometry definition\n"
        "    in the XPL file.\n"
    ;

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
