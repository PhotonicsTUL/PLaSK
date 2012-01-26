#ifndef PLASK__MODULE_H
#define PLASK__MODULE_H

/** @file
This file includes base class for modules.
@see @ref modules
*/

/** @page modules Modules

@section modules_about About modules
Modules in PLaSK are calculations units. Each is represented by subclass of plask::Module.

Each module has one or more input and one or more output realized by @ref providers "providers and receivers mechanism".
This allows for communication between modules.

Typically, each module is connected with @ref geometry "geometry element" which describe physical property (mainly materials) of laser or its fragment.
This @ref geometry "geometry element" defines (local) calculation space for module.

Typically, each module also includes @ref meshes "mesh" which represent set of point (in calculation space) in which module calculate its result.
If another module requests for data in points other than these includes in mesh, result can be @ref interpolation "interpolated".

@section modules_use How modules are typically used?
TODO

Note that typically, modules are used from python scripts.

@section modules_write How to write new a calculation module?
To write module you should:
-# Write new class which inherit from plask::Module.
-# Implement plask::Module::getName method. This method should just return name of your module.
-# Place in your class public providers and receivers fields:
    - providers fields allows your module to provide results to another modules or to reports,
    - receivers field are used to getting data which are required for calculations from another modules (precisely, from its providers),
    - you don't have to care about connection between providers and corresponding receivers (this is done externally),
    - more details can be found in @ref providers.
-# Typically, implement calculate method. This method is a place for your calculation code. You don't have to implement it if you don't need to do any calculations.
-# Optionally implement plask::Module::getDescription method. This method should just return description of your module.

Example:
@code
#include <plask/plask.hpp>

struct MyModule: public plask::Module {
    
    ReceiverAType a;
    
    ReceiverBType b;
    
    ProviderCType c;
    
    virtual std::string getName() const {
        return "My module name";
    }
    
    virtual std::string getDescription() const {
        return "Calculate c using a and b.";
    }
    
    void calculateC() {
        if (!a.changed && !b.changed)   //if input doesn't changed after last calculation
            return;                     //we don't have to update c
        //...here calculate c...
        //values of a and b can be get by a() and b()
        
        //say receivers of c that it has been changed:
        c.fireChanged();
    }
    
};
@endcode
*/

#include <string>

namespace plask {

/**
 * Base class for all modules.
 * 
 * @see @ref modules
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

    /*
     * Make calculations. Place for calculation code in inherited classes.
     *
     * Can throw exception in case of errors.
     *
     * By default do nothing.
     */
    //virtual void calculate() {}

};

}       //namespace plask

#endif // PLASK__MODULE_H
