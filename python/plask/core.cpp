#include <plask/exceptions.h>
#include <plask/space.h>
#include <plask/module.h>

#include "globals.h"
#include <numpy/arrayobject.h>
using namespace plask::python;

// Declare some initialization functions
namespace plask { namespace python {

void initMaterials();
void initGeometry();

void register_vector();
void register_mesh();
void register_providers();

// Config
Config config;
bool Config::z_up = true;

void register_space() {
    py::object space_module { py::handle<>(py::borrowed(PyImport_AddModule("plask.space"))) };
    py::scope().attr("space") = space_module;
    py::scope scope = space_module;

    py::class_<plask::space::Cartesian2d> spacexy("Cartesian2D",
        "Cartesian two-dimensional space. The structure is assumed to be uniform in the third direction.");
    spacexy.attr("DIMS") = int(plask::space::Cartesian2d::DIMS);

    py::class_<plask::space::Cylindrical2d> spacerz("Cylindrical2D",
        "Cyllindrical two-dimensional space. The structure is assumed to have cyllindrical symmetry.");
    spacerz.attr("DIMS") = int(plask::space::Cylindrical2d::DIMS);

    py::class_<plask::space::Cartesian3d> spacexyz("Cartesian3D",
        "Cartesian three-dimensional space.");
    spacexyz.attr("DIMS") = int(plask::space::Cartesian3d::DIMS);

}

}}



BOOST_PYTHON_MODULE(plaskcore)
{
    // Initialize numpy
    import_array();

    py::scope scope; // Default scope

    // Config
    register_config();

    // Vectors
    register_vector();


    // Space
    register_space();

    // Meshes
    register_mesh();

    // Init subpackages
    initMaterials();
    initGeometry();

    // Modules
    py::class_<plask::Module, plask::shared_ptr<plask::Module>, boost::noncopyable>("Module", "Base class for all modules", py::no_init)
        .add_property("name", &plask::Module::getName, "Full name of the module")
        .add_property("description", &plask::Module::getDescription, "Short description of the module")
    ;

    // Exceptions
    register_exception<plask::NotImplemented>(PyExc_NotImplementedError);

    // PLaSK version
    scope.attr("version") = PLASK_VERSION;
    scope.attr("version_major") = PLASK_VERSION_MAJOR;
    scope.attr("version_minor") = PLASK_VERSION_MINOR;
}
