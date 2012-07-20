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

@section modules_writing How to write a new calculation module?
To write module you should:
-# If you write an official module for PLaSK, create a subdirectory streucture in directory modules. The subdirectories
   in modules should have a form \b modules/category/your_module, e.g. \b modules/optical/FDTD.
   It is also possible to write fully external modules by PLaSK users, using the descpription below (TODO).
-# Copy modules/skel/CMakeLists.txt from to your subdirectory. You may edit the copied file to suit your needs.
-# By default all the sources should be placed flat in your module subdirectory and the Python interface in the subdirectory
   \b your_module/python.

Once you have your source tree set up, do the following:
-# Write new class which inherit from plask::ModuleOver<SPACE_TYPE> or plask::ModuleWithMesh<SPACE_TYPE, MESH_TYPE>.
   SPACE_TYPE should be one of Geometry2DCartesian, Geometry2DCylindrical, or Geometry3D and should indicate in what space your module
   is doing calculations. If you want to allow user to specify a mesh for your module, inherit from plask::ModuleWithMesh<SPACE_TYPE, MESH_TYPE>,
   where MESH_TYPE is the type of your mesh.
-# Implement plask::Module::getName method. This method should just return name of your module.
-# Implement plask::Module::onInitialize and optionally plask::Module::onInvalidate methods.
-# Place in your class public providers and receivers fields:
    - provider fields allow your module to provide results to another modules or to reports,
    - receiver fields are used to getting data which are required for calculations from another modules
      (precisely, from its providers),
    - you don't have to care about connection between providers and corresponding receivers (this is done externally),
    - note that most providers are classes obtained by using plask::ProviderFor template,
    - more details can be found in @ref providers.
-# If you need boundary conditions, place in your class public plask::BoundaryConditions fields which are containers of boundary-condition pairs.
-# Typically, implement calculation method. This method is a place for your calculation code.
   You don't have to implement it if you don't need to do any calculations. You can also write more methods performing
   different calculations, however, you need to clearly document them. Each calculation method must call plask::Module::initCalculation()
   as the first operation.
-# Optionally implement plask::Module::getDescription method. This method should just return description of your module.
-# Write the Python interface to your class using Boost Python. See the Boos Python documentation or take a look into
   modules/skel/python/module.cpp (for your convenience we have provided some macros that will faciliate creation
   of Python interface).
-# (TODO: in future do something to make the module available in GUI)
-# Finally write a good user manual for your module ;)


\subsection modules_writing_details Writing modules in depth

Below we explain the above steps in detail on a simple example. When following this tutorial work already on your own module, or
write a sample code in a separate directory in order to keep the PLaSK source tree (and repository) clean.

Assume that we want to write a module computing a waveguide effective index and optical field intensity for edge emitting lasers.
The module performs its computation using finite differences method. Hence, we name it \b FiniteDifferencesModule.

To start we create a subdirectory \a modules/optical/finite_diff under the PLaSK trunk write a file \a finite_differences.h to it.
Also we need to copy the \a CMakeLists.txt file from \a modules/skel to our directory. In most cases we will not need to edit this
file, unless our modules uses some external libraries. If for example, our module uses LAPACK, the \a CMakeLists.txt should look
like this (In the below example all comments are skipped):

\verbatim
include(${CMAKE_SOURCE_DIR}/cmake/modules.cmake)

find_package(LAPACK)
set(MODULE_LINK_LIBRARIES ${LAPACK_LIBRARIES})

make_default()
\endverbatim

If you are working within the PLaSK source, remember to add all the directories and files to the subversion repository
(only for your real module naturally)!

\subsubsection modules_writing_tutorial Module C++ class

Now we assume that our module uses rectilinear mesh provided by the user, so we inherit our class from
\link plask::ModuleWithMesh plask::ModuleWithMesh<plask::Geometry2DCartesian, plask::RectilinearMesh2D>\endlink. So our header
\a finite_differences.h should begin as follows:

\code
#ifndef PLASK__MODULES__OPTICAL__FINITE_DIFFERENCES_H
#define PLASK__MODULES__OPTICAL__FINITE_DIFFERENCES_H

#include <plask/plask.hpp>

namespace plask { namespace modules { namespace optical_finite_differences { // put everything in private namespace

class FiniteDifferencesModule: public plask::ModuleWithMesh < plask::Geometry2DCartesian, plask::RectilinearMesh2D >
{
\endcode

Then, you declare all the fields and methods of the class. We will skip all the fields that are required privately for
computations from this tutorial and focus only on the ones necessary for PLaSK interface. Assume, that \b FiniteDifferencesModule
needs a temperature distribution and wavelength as an input, and outputs effective index of the waveguide and the optical
field intensity.
Additionaly, boundary conditions of the first kind on temperature is needed.
Hence, we declare the following \link providers providers, receivers\endlink and \link boundaries boundary conditions\endlink:

\code
  public:

    plask::ReceiverFor<plask::Wavelength> inWavelength;

    plask::ReceiverFor<plask::Temperature, plask::Geometry2DCartesian> inTemperature;

    plask::ProviderFor<plask::EffectiveIndex>::WithValue outNeff;

    plask::ProviderFor<plask::OpticalIntensity, plask::Geometry2DCartesian>::Delegate outIntensity;

    plask::BoundaryConditions<plask::RectilinearMesh2D, double> boundaryConditionsOnTemperature;
\endcode

In the code above, we have declared two receivers (by convention in PLaSK, names of every receiver in all modules should begin with
\a in prefix). \c inWavelength can receive a single value either from some connected module (e.g. a computed gain maximum) or specified
by the user. On the other hand \c inTemperature receives temperature distribution, which is a scalar field. For this reason it is
necessary to specify the type of the space in which this distribution will be computed. Naturally it should match the working space of your
module.

When declaring providers, one needs to specify how the provided value can be obtained. In our case \c outNeff (again, the names of every
provider in all modules should begin with \a out prefix) will hold its value internally i.e. you will be able to assign a value to it as simply
as <tt>outNeff = 3.5;</tt>. On the other hand, \c outIntensity is a delegate provider i.e. you will need to write a method which computes
the light intensity on demand (we will later show you how).

As your module inherits plask::ModuleWithMesh, there is already a shared pointer to the \c geometry and \c mesh available. However, it might be
a good idea to create the class member field for the light intensity computed on the module mesh to be able to efficiently provide it to other
modules. You can use any type for array for this (e.g. std::vector<double>), but—as the modules exchange field data using plask::DataVector
class—it is best to use this class for internal storage as well (it behaves mostly like std::vector, with some additional improvements required
for shared memory management between the modules). So we declare the private field:

\code
  private:

    plask::DataVector<double> computed_light_intensity;
\endcode

Now, you can write the constructor to your class. By convention this constructor should take no arguments as all the module configuration
parameters must be able to be set by the user afterwards (in future you might want to create a configuration window for your module for GUI).

\code
  public:

    FiniteDifferencesModule():
        outIntensity(this, &FiniteDifferencesModule::getIntensity) // attach the method returning the light intensity to delegate provider
    {
        inTemperature = 300.;                                      // set default value for input temperature
    }
\endcode

In the above illustration, we initialize the \c outIntensity provider with the pointer to the module itself and the address of the method
computing the light intensity (we write this method \ref module_delegate_provider_method "later"). Also, we set the default value of the
temperature to 300&nbsp;K in the whole structure. As there is not default value for inWavelenght, the user will have to provide it
(either manually or from some wavelength provider) or the exception will be raised when we try to retrieve the wavelength value in our
computation method.

Before we write a computation method (or several computation methods), we must realize that the module can be in two states: initialized or invalidated.
When the user creates the module object, sets the geometry and the mesh, there is still no memory allocated for our results (i.e. \c computed_light_intensity
is an empty vector) nor the results are know. This is called an invalidated state. When in this state, the first step before any calculation is to allocate
the memory: resize the \c computed_light_intensity to the proper size, allocate some internal matrices (not mentioned in this tutorial), etc. The size
of the allocated memory depends e.g. on the mesh size, so it cannot be done in the constructor. For this reason, in the beginning of every calculation
function you should call the method \link plask::Module::initCalculation initCalculation\endlink(), which is defined in plask::Module base class. If the
module has been in invalidated state, this it will call virtual method \b onInitialize(), in which you can put all your initialization code. Then the module
will be put in initialized state (hence, subsequent calls to computational method will not call \c onInitialize() unless module is forcefully invalidated
(see \ref module_invalidate "below"). The code on the initialization method may look like this:

\code
  protected:

    virtual void onInitialize()
    {
        if (!geometry) throw NoGeometryException(getId());  // test if the user has provided geometry
        if (!mesh) throw NoMeshException(getId());          // test if the user has provided the mesh
        // do any memory allocations or your internal matrices, etc.
    }
\endcode

\anchor module_invalidate Even after some computations have been performed, the user might change the geometry of the structure or the mesh. In such case
the results of your computations becomes outdated and the sizes of some matrices might need to change. In such situation, the module is put back to invalidated
state and, similarly to the initialization, the virtual method \b onInvalidate() is called. This method may look like this:

\code
    virtual void onInvalidate()
    {
        outNeff.invalidate();               // clear the computed value
        computed_light_intensity.reset();   // free the light intensity array
        // you may free the memory allocated in onInitialize unless you use some containers with automatic memory management
    }
\endcode

This method can be also forcefully called by the user issuing <tt>your_module.invalidate();</tt> command. This might be done in order to free them
memory for some other computations. For this reason you should free all large chunks of memory in \c onInvalidate(). However, If you store your data
in an array with no built-in memory management (e.g. old-style C-arrays), you have to check if the memory has been previously allocated, as \c onInvalidate()
might be called before \c onInitialize(). Furthermore you should call this method from your class destructor in such case.

After the module has been invalidated, before the next computations \c onInitialize() will be called, so the new memory will be allocated as needed.

Now you can write your core computational methods. Just remember to call \link plask::Module::initCalculation initCalculation\endlink() in the beginning.
Furthermore, if the provided values change (and they probably do), you must call method \link plask::Provider::fireChanged fireChanged\endlink() for each
of your providers, to notify the connected receivers about the change. Here is the sample code:

\code
  public:

      void compute()
      {
          initCalculation(); // calls onInitialize if necessary and puts module in initialized state

          computed_light_intensity.reset(); // clear the previously computed light intensity (if any)

          DataVector<double> temperature = inTemperature(*mesh); // gets temperature computed by some other module
          double wavelenght = inWavelenght();                    // gets the wavelength

          // [...] perform your calculations

          outNeff = new_computed_effective_index;
          outNeff.fireChanged();            // inform others we computed new effective index
          outIntensity.fireChanged();       // we will also be providing new light intensity
      }
\endcode

Assume that in the above sample computation method, we did not compute the light intensity. We will do it only if someone wants to know it.
Of course the choice when to compute what depends strongly on your particular module and you should focus on efficiency (i.e. compute only
what is necessary) when making decisions on this matter.

\anchor module_delegate_provider_method The last thing to do is to write the method called by the delegate provider \c outIntensity when
someone wants to get the optical field intensity distribution. The arguments and return value of this method depend on the provider type.
For interpolated fields they will look like in the following example:

\code
  protected:

    const DataVector<double> getIntensity(const plask::MeshD<2>& destination_mesh, plask::InterpolationMethod interpolation_method=DEFAULT_INTERPOLATION)
    {
        if (!outNeff.hasValue()) throw NoValue(OpticalIntensity::NAME); // this is one possible indication that the module is in invalidated state

        if (computed_light_intensity.size() == 0)    // we need to compute the light intensity
        {
            computed_light_intensity.reset(mesh.size()); // allocate space for the light intensity
            /// [...] compute light intensity
        }

        if (interpolation_method == DEFAULT_INTERPOLATION) interpolation_method = INTERPOLATION_LINEAR;
        return interpolate(*mesh, computed_light_intensity, destination_mesh, interpolation_method); // interpolate your data to the requested mesh
    }
\endcode

The important elements of the above method are the first and the last lines. In the former one, we check if the computations have been performed
and are up-to-date (remember, we have cleared the value of \c outNeff in \c onInvalidate()). Otherwise we throw an exception. In the last line
we use plask::interpolate function to interpolate our data to the receiver mesh (which is provided as \c destination_mesh argument).

You can now finish your class definition:

\code
}; // class FiniteDifferencesModule

}}} // namespace plask::modules::optical_finite_differences

#endif // PLASK__MODULES__OPTICAL_FINITE_DIFFERENCES_H
\endcode


\subsubsection module_python_interface_tutorial Writing Python interface

Once you have written all the C++ code of your module, you should export it to the Python interface. To do this, create a subdirectory
named \a python in your module tree (i.e. \a modules/optical/finite_diff/python) and copy \a your_module.cpp from \a modules/skel/python
there (changing its name e.g. to \a finite_differences_python.cpp).

Then edit the file to export all necessary methods and public fields. The contents of this file should look more less like this:

\code
#include <cmath>
#include <plask/python.hpp>
using namespace plask;
using namespace plask::python;

#include "../finite_differences.h"
using namespace plask::modules::optical_finite_differences; // use your module pricate namespace

BOOST_PYTHON_MODULE(fd)
{
    {CLASS(FiniteDifferencesModule, "FiniteDifferencesCartesian2D",
        "Calculate optical modes and optical field distribution using the finite\n"
        "differences method in Cartesian two-dimensional space.")

     RECEIVER(inWavelength, "Wavelength of the light");

     RECEIVER(inTemperature, "Temperature distribution in the structure");

     PROVIDER(outNeff, "Effective index of the last computed mode");

     PROVIDER(outIntensity, "Light intensity of the last computed mode");

     METHOD(compute, "Perform the computations");
    }
}
\endcode

\c BOOST_PYTHON_MODULE macro takes the name of the package with your module (without quotation marks). Your module will be accessible to
Python if the user imports this package as:

\code{.py}
import plask.optical.fd
\endcode

The arguments of the \c CLASS macro are your module class, the name in which it will appear in Python, and short module documentation
(mind the braces outside of the \c CLASS: they are important if you want to put more than one module in a single interface file, so they
will appear in a single package).

Next you define your exported class member fields, properties (fake fields, which call your class methods on reading or assignment), methods,
providers, and receivers (you have probably noticed that providers and receivers are just class member fields, but they need to be exported
using separate macros, due to some additional logic necessary). Below, there is a complete list of macros exporting class elements and we believe
it is self-explanatory:

\code
METHOD(method_name, "Short documentation", "name_or_argument_1", arg("name_of_argument_2")=default_value_of_arg_2, ...);

RO_FIELD(field_name, "Short documentation");                                                    // read-only field

RW_FIELD(field_name, "Short documentation");                                                    // read-write field

RO_PROPERTY(python_property_name, get_method_name, "Short documentation");                      // read-only property

RW_PROPERTY(python_property_name, get_method_name, set_method_name, "Short documentation");     // read-write property

RECEIVER(inReceiver, "Short documentation");                                                    // receiver in the module

PROVIDER(outProvider, "Short documentation");                                                   // provider in the module
\endcode

When defining methods that take some arguments, you need to put their names after the documentation, so the user can set
them by name, as for typical Python methods. You can also specify some default values of some arguments as shown above.

After you successfully compile the module and the Python interface (just run \c cmake and \c make), the user should be able to execute
the following Python code:

\code{.py}
import numpy
import plask
import plask.optical.fd

module = plask.optical.fd.FiniteDifferencesCartesian2D()
module.geometry = plask.geometry.Geometry2DCartesian(plask.geometry.Rectangle(2, 1, "GaN"))
module.mesh = plask.mesh.RectilinearMesh2D(numpy.linspace(0,2), numpy.linspace(0,1))
module.inTemperature = 280
module.compute()
print(module.outNeff())
\endcode

\subsubsection module_testing_tutorial Writing automatic module tests

TODO


This concludes our short tutorial. Now you can go on and write your own calculation module. Good luck!


*/

#include <string>

#include "log/log.h"
#include "log/data.h"
#include "mesh/mesh.h"
#include "geometry/space.h"
#include "geometry/reader.h"

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
     * Initialize the module.
     * Default implementation just does nothing, however it is a good idea to overwrite it in subclasses and put initialization code in it.
     */
    virtual void onInitialize() {}

    /**
     * Begin calculations.
     * This method is called ALWAYS from initCalculation(). You can put some code common for all your calculation methods here
     * \param fresh indicates whether the module has been just switched from uninitialized state
     */
    virtual void onBeginCalculation(bool fresh) {}

    /**
     * This method is called by invalidate() to reset stored values.
     *
     * Default implementation does nothing.
     * @see invalidate()
     */
    virtual void onInvalidate() {}

    /**
     * This should be called on beginning of each calculation method to ensure that module will be initialized.
     * It's does nothing if module is already initialized and calls init() if it's not.
     */
    void initCalculation() {
        if (!initialized) {
            writelog(LOG_INFO, "Initializing module");
            onInitialize();
            initialized = true;
            onBeginCalculation(true);
        } else {
            onBeginCalculation(false);
        }
    }

  public:

    Module(): initialized(false) {}

    /// Virtual destructor (for subclassing). Do nothing.
    virtual ~Module() {}

    /**
     * Load configuration from given @p config.
     *
     * XML reader (source field) of @p config point to opening of @c this module tag and
     * after return from this method should point to this module closing tag.
     *
     * Default implementation require empty configuration (just call <code>conf.source.requireTagEnd();</code>).
     * @param config source of configuration
     */
    virtual void loadConfiguration(GeometryReader& source);

    /**
     * Check if module is already initialized.
     * @return @c true only if module is already initialized
     */
    bool isInitialized() { return initialized; }

    /**
     * This method should be and is called if something important was changed: calculation space, mesh, etc.
     *
     * Default implementation set initialization flag to @c false and can call onInvalidate() if initialization flag was @c true.
     * @see onInitialize()
     */
    void invalidate() {
        if (initialized) {
            initialized = false;
            writelog(LOG_INFO, "Invalidating module");
            onInvalidate();
        }
    }

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
    Data2DLog<ArgT, ValT> dataLog(const std::string& chart_name, const std::string& axis_arg_name, const std::string& axis_val_name) {
        return Data2DLog<ArgT, ValT>(getId(), chart_name, axis_arg_name, axis_val_name);
    }


    template<typename ArgT = double, typename ValT = double>
    Data2DLog<ArgT, ValT> dataLog(const std::string& axis_arg_name, const std::string& axis_val_name) {
        return Data2DLog<ArgT, ValT>(getId(), axis_arg_name, axis_val_name);
    }

    /**
    * Log a message for this module
    * \param level log level to log
    * \param msg log message
    * \param params parameters passed to format
    **/
    template<typename ...Args>
    void writelog(LogLevel level, std::string msg, Args&&... params) { plask::writelog(level, getId() + ": " + msg, std::forward<Args>(params)...); }

};

/**
 * Base class for all modules operating on specified space
 */
template <typename SpaceT>
class ModuleOver: public Module {

    void diconnectGeometry() {
        if (this->geometry)
            this->geometry->changedDisconnectMethod(this, &ModuleOver<SpaceT>::onGeometryChange);
    }

  protected:

    /// Space in which the calculations are performed
    shared_ptr<SpaceT> geometry;

  public:

    /// of the space for this module
    typedef SpaceT SpaceType;

    ~ModuleOver() {
        diconnectGeometry();
    }

    /**
     * This method is called when calculation space (geometry) was changed.
     * It's just call invalidate(); but subclasses can customize it.
     * @param evt information about calculation space changes
     */
    virtual void onGeometryChange(const Geometry::Event& evt) {
        this->invalidate();
    }


    /**
     * Get current module geometry space.
     * @return current module geometry space
     */
    inline shared_ptr<SpaceT> getGeometry() const { return geometry; }

    /**
     * Set new geometry for the module
     * @param geometry new geometry space
     */
    void setGeometry(const shared_ptr<SpaceT>& geometry) {
        if (geometry == this->geometry) return;
        writelog(LOG_INFO, "Attaching geometry to the module");
        diconnectGeometry();
        this->geometry = geometry;
        if (this->geometry)
            this->geometry->changedConnectMethod(this, &ModuleOver<SpaceT>::onGeometryChange);
        initialized = false;
    }
};

/**
 * Base class for all modules operating on specified olding an external mesh
 */
template <typename SpaceT, typename MeshT>
class ModuleWithMesh: public ModuleOver<SpaceT> {

    void diconnectMesh() {
        if (this->mesh)
            this->mesh->changedDisconnectMethod(this, &ModuleWithMesh<SpaceT, MeshT>::onMeshChange);
    }

  protected:

    /// Mesh over which the calculations are performed
    shared_ptr<MeshT> mesh;

  public:

    /// Type of the mesh for this module
    typedef MeshT MeshType;

    ~ModuleWithMesh() {
        diconnectMesh();
    }

    /**
     * This method is called when mesh was changed.
     * It's just call invalidate(); but subclasses can customize it.
     * @param evt information about mesh changes
     */
    virtual void onMeshChange(const typename MeshT::Event& evt) {
        this->invalidate();
    }

    /**
     * Get current module mesh.
     * @return current module mesh
     */
    inline shared_ptr<MeshT> getMesh() const { return mesh; }

    /**
     * Set new mesh for the module
     * @param mesh new mesh
     */
    void setMesh(const shared_ptr<MeshT>& mesh) {
        if (mesh == this->mesh) return;
        this->writelog(LOG_INFO, "Attaching mesh to the module");
        diconnectMesh();
        this->mesh = mesh;
        if (this->mesh)
            this->mesh->changedConnectMethod(this, &ModuleWithMesh<SpaceT, MeshT>::onMeshChange);
        this->initialized = false;
    }

    /**
     * Set new mesh got from generator
     * \param generator mesh generator
     */
    void setMesh(MeshGeneratorOf<MeshT>& generator) {
        setMesh(generator(this->geometry->getChild()));
    }
};


}       //namespace plask

#endif // PLASK__MODULE_H
