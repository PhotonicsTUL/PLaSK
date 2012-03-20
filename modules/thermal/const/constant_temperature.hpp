#ifndef PLASK__CONSTANT_TEMPERATURE_HPP
#define PLASK__CONSTANT_TEMPERATURE_HPP

/** @file
This file includes module which provide constant temperature.
*/

#include <plask/plask.hpp>

namespace plask { namespace const_temp {

/**
 * Module which provide constant temperature in all space.
 */
struct ConstantTemperatureModule: public Module {

    virtual std::string getName() const {
        return "Thermal: Constant Temperature";
    }

    virtual std::string getDescription() const {
        return "Return a constant temperature in all spaces";
    }

};

}}       //namespace plask::const_temp

#endif // PLASK__CONSTANT_TEMPERATURE_HPP
