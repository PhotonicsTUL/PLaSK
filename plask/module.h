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
#include "geometry/calculation_space.h"

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

    template<typename ...Args>
    void log(LogLevel level, Args&&... params) {}

};

/**
 * Base class for all modules operating on two-dimensional Cartesian space
 */
class ModuleCartesian2d: public Module {

  protected:

    /// Space in which the calculations are performed
    shared_ptr<Space2dCartesian> geometry;

  public:

    ModuleCartesian2d(const shared_ptr<Space2dCartesian>& geometry) : geometry(geometry) {}

    ModuleCartesian2d() = default;

    /**
     * Set new geometry for the module
     *
     * @param new_geometry new geometry space
     **/
    void setGeometry(const shared_ptr<Space2dCartesian>& new_geometry) {
        log(LOG_INFO, "Attaching new geometry to the module.");
        geometry = new_geometry;
    }

    // Get current module geometry space
    inline shared_ptr<Space2dCartesian> getGeometry() const { return geometry; }
};

}       //namespace plask

#endif // PLASK__MODULE_H
