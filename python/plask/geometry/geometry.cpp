#include <cmath>
#include "../python_globals.h"
#include <boost/python/stl_iterator.hpp>
#include "../python_util/raw_constructor.h"

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
        u8"PLaSK geometry classes.\n\n"

        u8"Classes and functions defined in this module can be used to create and modify\n"
        u8"geometry description in PLaSK. See :ref:`sec-geometry-python` for more details.\n\n"

        u8"Example:\n"
        u8"    To create a simple stack with two identical rectangles and check its total\n"
        u8"    size, use the following commands:\n\n"

        u8"    >>> rectangle = geometry.Block2D(4, 2, 'GaAs')\n"
        u8"    >>> stack = geometry.Stack2D()\n"
        u8"    >>> stack.prepend(rectangle)\n"
        u8"    <plask.geometry.PathHint at 0x40a52f8>\n"
        u8"    >>> stack.prepend(rectangle)\n"
        u8"    <plask.geometry.PathHint at 0x40a50d8>\n"
        u8"    >>> stack.bbox\n"
        u8"    plask.geometry.Box2D(0, 0, 4, 4)\n\n"

        u8"    Now, to create a Cartesian two-dimensional geometry over it:\n\n"

        u8"    >>> geometry.Cartesian2D(stack)\n"
        u8"    <plask.geometry.Cartesian2D object at (0x571acd0)>\n\n"

        u8"    You may also modify any existing geometry object:\n\n"

        u8"    >>> rectangle.height = 3\n"
        u8"    >>> stack.bbox\n"
        u8"    plask.geometry.Box2D(0, 0, 4, 6)\n\n"

        u8"See also:\n"
        u8"    Section :ref:`sec-geometry` for detailed information of geometry.\n\n"

        u8"    XPL section :xml:tag:`geometry` for reference of the geometry definition\n"
        u8"    in the XPL file.\n"
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
