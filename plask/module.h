#ifndef PLASK__MODULE_H
#define PLASK__MODULE_H

/** @page modules Modules
 * @section about About modules
 * @section modules_write How to write new a calculation module?
 * To write module you should:
 * -# Write new class which inherit from plask::Module.
 * -# Implement plask::Module::getName method. This method should just return name of your module.
 * -# Place in your class public providers and receivers fields:
 *    - providers fields allows your module to provide results to another modules or to reports,
 *    - receivers field are used to getting data which are required for calculations from another modules (precisely, from its providers),
 *    - you don't have to care about connection between providers and corresponding receivers (this is done externally),
 *    - more details can be found in @ref providers.
 * -# Typically, implement plask::Module::calculate method. This method is a place for your calculation code. You don't have to implement it if you don't need to do any calculations.
 * -# Optionally implement plask::Module::getDescription method. This method should just return description of your module.
 */

#include <string>

namespace plask {

/**
 * Base class for all modules.
 */
struct Module {

    /**
     * Do nothing.
     */
    virtual ~Module() {}

    /**
     * @return name of this module
     */
    virtual std::string getName() const = 0;

    /**
     * @return description of this module (empty string by default)
     */
    virtual std::string getDescription() const { return ""; }

    /**
     * Make calculations. Place for calculation code in inherited classes.
     *
     * Can throw exception in case of errors.
     *
     * By default do nothing.
     */
    virtual void calculate() {}

};

}       //namespace plask

#endif // PLASK__MODULE_H
