#include <cmath>
#include <boost/python.hpp>
namespace py = boost::python;

#include "../constant_temperature.h"

BOOST_PYTHON_MODULE(const)
{
    py::class_<plask::ConstantTemperatureModule, plask::shared_ptr<plask::ConstantTemperatureModule>, py::bases<plask::Module>>("ConstantTemperature",
        "Module providing constant temperature");
}
