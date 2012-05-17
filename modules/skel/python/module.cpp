/** \file
 * Sample Python wrapper for your module.
 */
#include <cmath>
#include <plask/python.hpp>
using namespace plask;
using namespace plask::python;

#include "../your_module_class_header.hpp"
using namespace plask::modules::your_module;

/**
 * Initialization of your module class to Python
 *
 * The \a module_name should be changed to match the name of the directory with our module
 * (the one where you have put CMakeLists.txt). It will be visible from user interface under this name.
 */
BOOST_PYTHON_MODULE(module_name)
{
    {CLASS(Class_Name, "YourModule", "Short module description and constructor documentation.")
        METHOD(method_name, "Short documentation", "name_or_argument_1", arg("name_of_argument_2")=default_value_of_arg_2, ...);
        RO_FIELD(field_name, "Short documentation"); // read-only field
        RW_FIELD(field_name, "Short documentation"); // read-write field
        RO_PROPERTY(python_property_name, get_method_name, "Short documentation"); // read-only property
        RW_PROPERTY(python_property_name, get_method_name, set_method_name, "Short documentation"); // read-write property
        // TODO Providers and Receivers
    }


}

