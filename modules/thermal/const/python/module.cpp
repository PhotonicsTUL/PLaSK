#include <cmath>
#include <boost/python.hpp>
namespace py = boost::python;

#include "../constant_temperature.hpp"

BOOST_PYTHON_MODULE(const)
{
    py::class_<plask::const_temp::ConstantTemperatureModule, plask::shared_ptr<plask::const_temp::ConstantTemperatureModule>, py::bases<plask::Module>>("ConstantTemperature",
        "Module providing constant temperature");
}
