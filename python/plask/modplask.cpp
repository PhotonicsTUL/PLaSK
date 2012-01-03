#include <boost/python.hpp>
namespace py = boost::python;

#include <config.h>
#include <plask/space.h>

#include "vector.h"

// Declare some initialization functions
namespace plask { namespace python {

void initMaterial();
void initGeometry();

}}

BOOST_PYTHON_MODULE(modplask)
{
    py::scope scope; // Default scope


    // Vectors
    plask::python::register_vector_h();


    // Space
    py::class_<plask::SpaceXY> spacexy("SpaceXY",
        "Cartesian two-dimensional space. The structure is assumed to be uniform in z-direction. "
        "The y-axis is perpendicular to epitaxial layers.");
    spacexy.attr("DIMS") = int(plask::SpaceXY::DIMS);

    py::class_<plask::SpaceRZ> spacerz("SpaceRZ",
        "Cyllindrical two-dimensional space. The structure is assumed to have cyllindrical symmetery. "
        "The axis of the cylinder (z-axis) is perpendicular to epitaxial layers.");
    spacerz.attr("DIMS") = int(plask::SpaceRZ::DIMS);

    py::class_<plask::SpaceXYZ> spacexyz("SpaceXYZ",
        "Cartesian three-dimensional space. Its z-axis is perpendicular to epitaxial layers.");
    spacexyz.attr("DIMS") = int(plask::SpaceXYZ::DIMS);


    // Init subpackages
    plask::python::initMaterial();
    plask::python::initGeometry();


    // PLaSK version
    scope.attr("version") = PLASK_VERSION;
    scope.attr("version_major") = PLASK_VERSION_MAJOR;
    scope.attr("version_minor") = PLASK_VERSION_MINOR;
}
