#include <plask/space.h>

#include "globals.h"
using namespace plask::python;

// Declare some initialization functions
namespace plask { namespace python {

void initMaterial();
void initGeometry();
void register_vector();

// Config
Config config;
bool Config::z_up = true;

}}

BOOST_PYTHON_MODULE(modplask)
{
    py::scope scope; // Default scope

    // Config
    register_config();

    // Vectors
    register_vector();


    // Space
    py::class_<plask::space::Cartesian2d> spacexy("SpaceCartesian2d",
        "Cartesian two-dimensional space. The structure is assumed to be uniform in z-direction. "
        "The y-axis is perpendicular to epitaxial layers.");
    spacexy.attr("DIMS") = int(plask::space::Cartesian2d::DIMS);

    py::class_<plask::space::Cylindrical2d> spacerz("SpaceCylindrical2d",
        "Cyllindrical two-dimensional space. The structure is assumed to have cyllindrical symmetery. "
        "The axis of the cylinder (z-axis) is perpendicular to epitaxial layers.");
    spacerz.attr("DIMS") = int(plask::space::Cylindrical2d::DIMS);

    py::class_<plask::space::Cartesian3d> spacexyz("SpaceCartesian3d",
        "Cartesian three-dimensional space. Its z-axis is perpendicular to epitaxial layers.");
    spacexyz.attr("DIMS") = int(plask::space::Cartesian3d::DIMS);


    // Init subpackages
    initMaterial();
    initGeometry();


    // PLaSK version
    scope.attr("version") = PLASK_VERSION;
    scope.attr("version_major") = PLASK_VERSION_MAJOR;
    scope.attr("version_minor") = PLASK_VERSION_MINOR;
}
