#include <boost/python.hpp>
namespace py = boost::python;

#include <config.h>

// Declare some initialization functions
namespace plask { namespace python {

void initMaterial();
void initGeometry();

}}

BOOST_PYTHON_MODULE(modplask)
{
    // Register plask exceptions

    py::scope scope; // Default scope

    // Init subpackages
    plask::python::initMaterial();
    plask::python::initGeometry();

    // PLaSK version
    scope.attr("version") = PLASK_VERSION;
    scope.attr("version_major") = PLASK_VERSION_MAJOR;
    scope.attr("version_minor") = PLASK_VERSION_MINOR;
}
