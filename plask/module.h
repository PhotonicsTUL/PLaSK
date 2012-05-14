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
-# If you write an official module for PLaSK, create a subdirectory streucture in directory modules. The subdirectories
   in modules should have a form \b modules/category/your_module, e.g. \b modules/optical/FDTD.
   It is also possible to write fully external modules by PLaSK users, using the descpription below (TODO).
-# Copy modules/skel/CMakeLists.txt from to your subdirectory. You may edit the copied file to suit your needs.
-# By default all the sources should be placed flat in your module subdirectory and the Python interface in the subdirectory
   \b your_module/python.

Once you have your source tree set up, do the following:
-# Write new class which inherit from plask::Module.
-# Implement plask::Module::getName method. This method should just return name of your module.
-# Place in your class public providers and receivers fields:
    - providers fields allows your module to provide results to another modules or to reports,
    - receivers field are used to getting data which are required for calculations from another modules
      (precisely, from its providers),
    - you don't have to care about connection between providers and corresponding receivers (this is done externally),
    - note that most providers are classes obtain by using plask::ProviderFor template,
    - more details can be found in @ref providers.
-# Typically, implement calculate method. This method is a place for your calculation code.
   You don't have to implement it if you don't need to do any calculations. You can also write more methods performing
   different calculations, however, you need to clearly document them.
-# Optionally implement plask::Module::getDescription method. This method should just return description of your module.
-# Finally write the Python interface to your class using Boost Python. See the Boos Python documentation or take a look into
   modules/skel/python/module.cpp (for your convenience we have provided some macros that will faciliate creation
   of Python interface).
-# (TODO: in future do something to make the module available in GUI)


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
        if (!a.changed && !b.changed)   // if input doesn't changed after last calculation
            return;                     // we don't have to update c
        // ...here calculate c...
        // values of a and b can be get by a() and b()

        // say receivers of c that it has been changed:
        c.fireChanged();
    }

};
@endcode
See also example in plask::Temperature description.
*/

#include <string>

#include "log/log.h"
#include "log/data.h"
#include "space.h"

namespace plask {

/**
 * Base class for all modules.
 *
 * @see @ref modules
 */
class Module {
protected:

    /// true only if module is initialized
    bool initialized;

    /**
     * Initialize module.
     * Defualt implementation just do nothing but it is good idea to overwrite it in subclasses and put initialization code in it.
     */
    virtual void init() {}

    /**
     * This should be called on beggining of each calculation method to ensure that module will be initialized.
     * It's do nothing if module is already initialized and call init() if it's not.
     */
    void beforeCalculation() {
        if (initialized) return;
        init();
        initialized = true;
    }

public:

    Module(): initialized(false) {}

    /**
     * Check if module is already initialized.
     * @return @c true only if module is already initialized
     */
    bool isInitialized() { return initialized; }

    /**
     * This method should be and is called if something important was changed: calculation space, mesh, etc.
     *
     * Default implementation set initialization flag to @c false.
     * @see beforeCalculation()
     */
    virtual void invalidate() { initialized = false; }

    /**
     * Do nothing.
     */
    virtual ~Module() {}

    /**
     * Get name of module.
     * @return name of this module
     */
    virtual std::string getName() const = 0;

    /**
     * Get module id (short name without white characters).
     *
     * Default implementation return the same as getName() but with all white characters replaced by '_'.
     * @return id of this module
     */
    virtual std::string getId() const;

    /**
     * Get a description of this module.
     * @return description of this module (empty string by default)
     */
    virtual std::string getDescription() const { return ""; }

    template<typename ArgT = double, typename ValT = double>
    Data2dLog<ArgT, ValT> dataLog(const std::string& chart_name, const std::string& axis_arg_name, const std::string& axis_val_name) {
        return Data2dLog<ArgT, ValT>(getId(), chart_name, axis_arg_name, axis_val_name);
    }

    template<typename ArgT = double, typename ValT = double>
    Data2dLog<ArgT, ValT> dataLog(const std::string& axis_arg_name, const std::string& axis_val_name) {
        return Data2dLog<ArgT, ValT>(getId(), axis_arg_name, axis_val_name);
    }

    /**
    * Log a message for this module
    * \param level log level to log
    * \param msg log message
    * \param params parameters passed to format
    **/
    template<typename ...Args>
    void log(LogLevel level, std::string msg, Args&&... params) { plask::log(level, getId() + ": " + msg, std::forward<Args>(params)...); }

};

/**
 * Base class for all modules operating on two-dimensional Cartesian space
 */
template <typename SpaceType>
class ModuleOver: public Module {

  protected:

    /// Space in which the calculations are performed
    shared_ptr<SpaceType> geometry;

  public:

    /**
     * Set new geometry for the module
     * @param new_geometry new geometry space
     */
    virtual void setGeometry(const shared_ptr<SpaceType>& new_geometry) {
        log(LOG_INFO, "Attaching geometry");
        geometry = new_geometry;
        //TODO attach listener which call invalidate on geometry changes
    }

    /**
     * Get current module geometry space.
     * @return current module geometry space
     */
    inline shared_ptr<SpaceType> getGeometry() const { return geometry; }
};

}       //namespace plask

#endif // PLASK__MODULE_H
