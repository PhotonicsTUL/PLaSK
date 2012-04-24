/** \file
 * Sample Python wrapper for your module.
 */
#include <cmath>
#include <boost/python.hpp>

#include "../eim.h"

using namespace plask::modules::eim;

/**
 * Initialization of your module to Python
 *
 * The \a module_name should be changed to match the name of the directory with our module
 * (the one where you have put CMakeLists.txt). It will be visible from user interface under this name.
 */
BOOST_PYTHON_MODULE(eim)
{
    ;
    //boost::python::class_<plask::YourModule, plask::shared_ptr<plask::YourModule>, py::bases<plask::Module>>("YourModule",
    //    "Short module description and constructor documentation.")
    //    .def("method", &YourModule::method, "Short documentation")
    //    .def_readonly("attribute", &YourModule::field, "Short documentation")
    //    .def_readwrite("attribute", &YourModule::field, "Short documentation")
    //    .add_attribute("attribute", &YourModule::getAttributeMethod, &YourModule::setAttributeMethod, "Shord documentation")
    // ;
}
