#ifndef PLASK__MODULE_H
#define PLASK__MODULE_H

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
     * Make calculations. Place for calculation code for subclasses.
     * 
     * Can throw exception in case of errors.
     * 
     * By default do nothing.
     */
    virtual void calculate() {}
    
};

}       //namespace plask

#endif // PLASK__MODULE_H
