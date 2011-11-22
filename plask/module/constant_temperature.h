#ifndef PLASK__CONSTANT_TEMPERATURE_H
#define PLASK__CONSTANT_TEMPERATURE_H

#include "../module.h"
#include "../provider/provider.h"

namespace plask {

/**
 * Module which provide constant temperature in all space.
 */
struct ConstantTemperatureModule: public Module {
    
    virtual std::string getName() const {
        return "constant temperature";
    }
    
    virtual std::string getDescription() const {
        return "return constant temperature in all space";
    }
    
};

}       //namespace plask

#endif // PLASK__CONSTANT_TEMPERATURE_H
