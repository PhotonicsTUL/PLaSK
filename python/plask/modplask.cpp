#include <string>

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


// Some config variables
struct Config {
    // Which axis is up (z or y)
    static bool z_up;
    static std::string get_vaxis() {
        if (z_up) return "z"; else return "y";
    }
    static void set_vaxis(std::string axis) {
        if (axis != "z" and axis != "y") {
            PyErr_SetString(PyExc_ValueError, "Only z or x allowed for vaxis");
            throw py::error_already_set();
        }
        z_up = axis == "z";
    }
};
bool Config::z_up = true;


BOOST_PYTHON_MODULE(modplask)
{
    py::scope scope; // Default scope


    // Config
    py::class_<Config, boost::noncopyable> config("config", "Global PLaSK configuration.", py::no_init);

    config.add_property("vaxis", &Config::get_vaxis, &Config::set_vaxis,
                             "Denotes orientation of coordinate system. Holds the name of an axis which is vertical, i.e. along layers growth direction.")
    ;



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
