#include <plask/exceptions.h>
#include <plask/module.h>

#include "python.hpp"
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

// Hack necessary as macro import_array wants to return some value
static inline bool plask_import_array() {
    import_array1(false);
    return true;
}

// Config
Config config;
AxisNames Config::axes = AxisNames::axisNamesRegister.get("xyz");

inline static void register_config()
{
    py::class_<Config>("config", "Global PLaSK configuration.", py::no_init)
        .def("__str__", &Config::__str__)
        .def("__repr__", &Config::__repr__)
        .add_property("axes", &Config::axes_name, &Config::set_axes,
                      "String representing axis names")
    ;
    py::scope().attr("config") = config;
}


}} // namespace plask::python

BOOST_PYTHON_MODULE(plaskcore)
{
    // Initialize numpy
    if (!plask_import_array()) throw(py::error_already_set());

    py::scope scope; // Default scope

    // Config
    register_config();

    // Vectors
    register_vector();

    // Materials
    initMaterials();

    // Geometry
    initGeometry();

    // Space
    register_calculation_spaces();

    // Meshes
    register_mesh();

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
