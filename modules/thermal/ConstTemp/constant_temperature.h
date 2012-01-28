#ifndef PLASK__CONSTANT_TEMPERATURE_H
#define PLASK__CONSTANT_TEMPERATURE_H

/** @file
This file includes module which provide constant temperature.
*/

#include <plask/plask.hpp>

namespace plask {

/**
 * Module which provide constant temperature in all space.
 */
struct ConstantTemperatureModule: public Module {

    virtual std::string getName() const {
        return "Constant Temperature";
    }

    virtual std::string getDescription() const {
        return "Return a constant temperature in all spaces";
    }

};

}       //namespace plask

#endif // PLASK__CONSTANT_TEMPERATURE_H
