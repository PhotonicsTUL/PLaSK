/** \file
 * Sample Python wrapper for your module.
 */
#include <boost/python.hpp>

// #include "../your_module_class_header.hpp"

using namespace plask::modules::your_module;

/**
 * Initialization of your module to Python
 *
 * The \a module_name should be changed to match the name of the directory with our module
 * (the one where you have put CMakeLists.txt). It will be visible from user interface under this name.
 */
BOOST_PYTHON_MODULE(module_name)
{
    //boost::python::class_<plask::YourModule, plask::shared_ptr<plask::YourModule>, py::bases<plask::Module>>("YourModule",
    //    "Short module description and constructor documentation.")
    //    .def("method", &YourModule::method, "Short documentation")
    //    .def_readonly("attribute", &YourModule::field, "Short documentation")
    //    .def_readwrite("attribute", &YourModule::field, "Short documentation")
    //    .add_attribute("attribute", &YourModule::getAttributeMethod, &YourModule::setAttributeMethod, "Shord documentation")
    // ;
}

