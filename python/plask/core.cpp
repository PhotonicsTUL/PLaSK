#include <plask/exceptions.h>
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
void register_calculation_spaces();

// Config
Config config;
AxisNames Config::axes = AxisNames::axisNamesRegister.get("xyz");

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

    // Geometry
    initGeometry();

    // Space
    register_calculation_spaces();

    // Meshes
    register_mesh();

    // Materials
    initMaterials();

    // Modules
    py::class_<plask::Module, plask::shared_ptr<plask::Module>, boost::noncopyable>("Module", "Base class for all modules", py::no_init)
        .add_property("name", &plask::Module::getName, "Full name of the module")
        .add_property("description", &plask::Module::getDescription, "Short description of the module")
    ;

    // Exceptions
    register_exception<plask::python::ValueError>(PyExc_ValueError);
    register_exception<plask::python::TypeError>(PyExc_TypeError);
    register_exception<plask::python::IndexError>(PyExc_IndexError);
    register_exception<plask::python::KeyError>(PyExc_KeyError);
    register_exception<plask::python::AttributeError>(PyExc_AttributeError);
    register_exception<plask::python::StopIteration>(PyExc_StopIteration);

    register_exception<plask::BadInput>(PyExc_ValueError);
    register_exception<plask::NotImplemented>(PyExc_NotImplementedError);


    // PLaSK version
    scope.attr("version") = PLASK_VERSION;
    scope.attr("version_major") = PLASK_VERSION_MAJOR;
    scope.attr("version_minor") = PLASK_VERSION_MINOR;
}
