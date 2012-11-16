#ifndef PLASK__SOLVER_H
#define PLASK__SOLVER_H

/** @file
This file includes base class for solvers.
@see @ref solvers
*/

/** @page solvers Solvers


@section solvers_about About solvers
Solvers in PLaSK are calculations units. Each is represented by subclass of plask::Solver.

Each solver has one or more input and one or more output realized by @ref providers "providers and receivers mechanism".
This allows for communication between solvers.

Typically, each solver is connected with @ref geometry "geometry object" which describe physical property (mainly materials) of laser or its fragment.
This @ref geometry "geometry object" defines (local) calculation space for solver.

Typically, each solver also includes @ref meshes "mesh" which represent set of point (in calculation space) in which solver calculate its result.
If another solver requests for data in points other than these includes in mesh, result can be @ref interpolation "interpolated".

@section solvers_use How solvers are typically used?
TODO

Note that typically, solvers are used from python scripts.

@section solvers_writing How to write a new calculation solver?
To write solver you should:
-# If you write an official solver for PLaSK, create a subdirectory streucture in directory solvers. The subdirectories
   in solvers should have a form \b solvers/category/your_solver, e.g. \b solvers/optical/FDTD.
   It is also possible to write fully external solvers by PLaSK users, using the descpription below (TODO).
-# Copy solvers/skel/CMakeLists.txt from to your subdirectory. You may edit the copied file to suit your needs.
-# By default all the sources should be placed flat in your solver subdirectory and the Python interface in the subdirectory
   \b your_solver/python.

Once you have your source tree set up, do the following:
-# Write new class which inherits from \link plask::SolverOver plask::SolverOver<SPACE_TYPE>\endlink or
   \link plask::SolverWithMesh plask::SolverWithMesh<SPACE_TYPE, MESH_TYPE>\endlink. SPACE_TYPE should be one of Geometry2DCartesian,
   Geometry2DCylindrical, or Geometry3D and should indicate in what space your solver is doing calculations. If you want to allow
   user to specify a mesh for your solver, inherit from \link plask::SolverWithMesh plask::SolverWithMesh<SPACE_TYPE, MESH_TYPE>\endlink,
   where MESH_TYPE is the type of your mesh.
-# Implement plask::Solver::getClassName method. This method should just return the pretty name of your solver class (the same you use in Python and XML).
-# Implement plask::Solver::onInitialize and optionally plask::Solver::onInvalidate methods.
-# Place in your class public providers and receivers fields:
    - provider fields allow your solver to provide results to another solvers or to reports,
    - receiver fields are used to getting data which are required for calculations from another solvers
      (precisely, from its providers),
    - you don't have to care about connection between providers and corresponding receivers (this is done externally),
    - note that most providers are classes obtained by using plask::ProviderFor template,
    - more details can be found in @ref providers.
-# If you need boundary conditions, place in your class public plask::BoundaryConditions fields which are containers of boundary-condition pairs.
-# Implement loadConfiguration method, which loads configuration of your solver from XML reader.
-# Typically, implement calculation method. This method is a place for your calculation code.
   You don't have to implement it if you don't need to do any calculations. You can also write more methods performing
   different calculations, however, you need to clearly document them. Each calculation method must call plask::Solver::initCalculation()
   as the first operation.
-# Optionally implement plask::Solver::getClassDescription method. This method should just return description of your solver.
-# Write the Python interface to your class using Boost Python. See the Boos Python documentation or take a look into
   \b solvers/skel/python/solver.cpp (for your convenience we have provided some macros that will faciliate creation
   of Python interface).
-# (TODO: in future do something to make the solver available in GUI)
-# Finally write a good user manual for your solver ;)


\subsection solvers_writing_details Writing solvers in depth

Below we explain the above steps in detail on a simple example. When following this tutorial work already on your own solver, or
write a sample code in a separate directory in order to keep the PLaSK source tree (and repository) clean.

Assume that we want to write a solver computing a waveguide effective index and optical field intensity for edge emitting lasers.
The solver performs its computation using finite differences method. Hence, we name it \b FiniteDifferencesSolver.

To start we create a subdirectory \a solvers/optical/finite_diff under the PLaSK trunk write a file \a finite_differences.h to it.
Also we need to copy the \a CMakeLists.txt file from \a solvers/skel to our directory. In most cases we will only need to edit
the line whith the command \c project in this file, unless our solvers uses some external libraries. If for example, our solver
uses LAPACK, the \a CMakeLists.txt should look like this (In the below example all comments are skipped):

The project name should math the pattern \a plask/solvergroup/solverlib, so in our case it will look like \a plask/optical/finite_diff.

\verbatim
project(plask/optical/finite_diff)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake)
find_package(PLaSK)

find_package(LAPACK)
set(SOLVER_LINK_LIBRARIES ${LAPACK_LIBRARIES})

make_default()
\endverbatim

If you are working within the PLaSK source, remember to add all the directories and files to the subversion repository
(only for your real solver naturally)!

\subsubsection solvers_writing_tutorial Solver C++ class

Now we assume that our solver uses rectilinear mesh provided by the user, so we inherit our class from
\link plask::SolverWithMesh plask::SolverWithMesh<plask::Geometry2DCartesian, plask::RectilinearMesh2D>\endlink. So our header
\a finite_differences.h should begin as follows:

\code
#ifndef PLASK__SOLVERS__OPTICAL__FINITE_DIFFERENCES_H
#define PLASK__SOLVERS__OPTICAL__FINITE_DIFFERENCES_H

#include <plask/plask.hpp>

namespace plask { namespace solvers { namespace optical_finite_differences { // put everything in private namespace

class FiniteDifferencesSolver: public plask::SolverWithMesh < plask::Geometry2DCartesian, plask::RectilinearMesh2D >
{
\endcode

Then, you declare all the fields and methods of the class. We will skip all the fields that are required privately for
computations from this tutorial and focus only on the ones necessary for PLaSK interface. Assume, that \b FiniteDifferencesSolver
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

    plask::BoundaryConditions<plask::RectilinearMesh2D, double> boundaryConditionsOnField;
\endcode

In the code above, we have declared two receivers (by convention in PLaSK, names of every receiver in all solvers should begin with
\a in prefix). \c inWavelength can receive a single value either from some connected solver (e.g. a computed gain maximum) or specified
by the user. On the other hand \c inTemperature receives temperature distribution, which is a scalar field. For this reason it is
necessary to specify the type of the space in which this distribution will be computed. Naturally it should match the working space of your
solver.

When declaring providers, one needs to specify how the provided value can be obtained. In our case \c outNeff (again, the names of every
provider in all solvers should begin with \a out prefix) will hold its value internally i.e. you will be able to assign a value to it as simply
as <tt>outNeff = 3.5;</tt>. On the other hand, \c outIntensity is a delegate provider i.e. you will need to write a method which computes
the light intensity on demand (we will later show you how).

As your solver inherits plask::SolverWithMesh, there is already a shared pointer to the \c geometry and \c mesh available. However, it might be
a good idea to create the class member field for the light intensity computed on the solver mesh to be able to efficiently provide it to other
solvers. You can use any type for array for this (e.g. std::vector<double>), but—as the solvers exchange field data using plask::DataVector
class—it is best to use this class for internal storage as well (it behaves mostly like std::vector, with some additional improvements required
for shared memory management between the solvers). So we declare the private field:

\code
  private:

    plask::DataVector<double> computed_light_intensity;
\endcode

Now, you can write the constructor to your class. By convention this constructor should take no configuration arguments as all the solver
configuration parameters must be able to be set by the user afterwards (in future you might want to create a configuration window for your
solver for GUI). The only constructor parameter is the name, which can be provided by user, and which should be passed to the parent class.
In addition, you should write getClassName method, which returns the category and the name of your solver class as seen by the end user
(it does not need to be the same as your real class name, but it should match the Python class name and solver name in XML file).

\code
  public:

    FiniteDifferencesSolver(const std::string name& name=""):
        plask::SolverWithMesh<plask::Geometry2DCartesian,plask::RectilinearMesh2D>(name),
        outIntensity(this, &FiniteDifferencesSolver::getIntensity) // attach the method returning the light intensity to delegate provider
    {
        inTemperature = 300.;                                      // set default value for input temperature
    }

    virtual std::string getClassName() const { return "optical.FiniteDifferencesCartesian2D"; }

\endcode

In the above illustration, we initialize the \c outIntensity provider with the pointer to the solver itself and the address of the method
computing the light intensity (we write this method \ref solver_delegate_provider_method "later"). Also, we set the default value of the
temperature to 300&nbsp;K in the whole structure. As there is not default value for inWavelength, the user will have to provide it
(either manually or from some wavelength provider) or the exception will be raised when we try to retrieve the wavelength value in our
computation method.

Before we write a computation method (or several computation methods), we must realize that the solver can be in two states: initialized or invalidated.
When the user creates the solver object, sets the geometry and the mesh, there is still no memory allocated for our results (i.e. \c computed_light_intensity
is an empty vector) nor the results are know. This is called an invalidated state. When in this state, the first step before any calculation is to allocate
the memory: resize the \c computed_light_intensity to the proper size, allocate some internal matrices (not mentioned in this tutorial), etc. The size
of the allocated memory depends e.g. on the mesh size, so it cannot be done in the constructor. For this reason, in the beginning of every calculation
function you should call the method \link plask::Solver::initCalculation initCalculation\endlink(), which is defined in plask::Solver base class. If the
solver has been in invalidated state, this it will call virtual method \b onInitialize(), in which you can put all your initialization code. Then the solver
will be put in initialized state (hence, subsequent calls to computational method will not call \c onInitialize() unless solver is forcefully invalidated
(see \ref solver_invalidate "below"). The code on the initialization method may look like this:

\code
  protected:

    virtual void onInitialize() {
        if (!geometry) throw NoGeometryException(getId());  // test if the user has provided geometry
        if (!mesh) throw NoMeshException(getId());          // test if the user has provided the mesh
        // do any memory allocations or your internal matrices, etc.
    }
\endcode

\anchor solver_invalidate Even after some computations have been performed, the user might change the geometry of the structure or the mesh. In such case
the results of your computations becomes outdated and the sizes of some matrices might need to change. In such situation, the solver is put back to invalidated
state and, similarly to the initialization, the virtual method \b onInvalidate() is called. This method may look like this:

\code
    virtual void onInvalidate() {
        outNeff.invalidate();               // clear the computed value
        computed_light_intensity.reset();   // free the light intensity array
        // you may free the memory allocated in onInitialize unless you use some containers with automatic memory management
    }
\endcode

This method can be also forcefully called by the user issuing <tt>your_solver.invalidate();</tt> command. This might be done in order to free them
memory for some other computations. For this reason you should free all large chunks of memory in \c onInvalidate(). However, If you store your data
in an array with no built-in memory management (e.g. old-style C-arrays), you have to check if the memory has been previously allocated, as \c onInvalidate()
might be called before \c onInitialize(). Furthermore you should call this method from your class destructor in such case.

After the solver has been invalidated, before the next computations \c onInitialize() will be called, so the new memory will be allocated as needed.

Now you can write your core computational methods. Just remember to call \link plask::Solver::initCalculation initCalculation\endlink() in the beginning.
Furthermore, if the provided values change (and they probably do), you must call method \link plask::Provider::fireChanged fireChanged\endlink() for each
of your providers, to notify the connected receivers about the change. Here is the sample code:

\code
  public:

      void compute() {
          initCalculation(); // calls onInitialize if necessary and puts solver in initialized state

          computed_light_intensity.reset(); // clear the previously computed light intensity (if any)

          DataVector<double> temperature = inTemperature(*mesh); // gets temperature computed by some other solver
          double wavelenght = inWavelength();                    // gets the wavelength

          // [...] perform your calculations

          outNeff = new_computed_effective_index;
          outNeff.fireChanged();            // inform others we computed new effective index
          outIntensity.fireChanged();       // we will also be providing new light intensity
      }
\endcode

Assume that in the above sample computation method, we did not compute the light intensity. We will do it only if someone wants to know it.
Of course the choice when to compute what depends strongly on your particular solver and you should focus on efficiency (i.e. compute only
what is necessary) when making decisions on this matter.

\anchor solver_delegate_provider_method The last thing to do is to write the method called by the delegate provider \c outIntensity when
someone wants to get the optical field intensity distribution. The arguments and return value of this method depend on the provider type.
For interpolated fields they will look like in the following example:

\code
  protected:

    DataVector<const double> getIntensity(const plask::MeshD<2>& destination_mesh, plask::InterpolationMethod interpolation_method=DEFAULT_INTERPOLATION) {
        if (!outNeff.hasValue()) throw NoValue(OpticalIntensity::NAME); // this is one possible indication that the solver is in invalidated state

        if (computed_light_intensity.size() == 0)    // we need to compute the light intensity
        {
            computed_light_intensity.reset(mesh.size()); // allocate space for the light intensity
            // [...] compute the light intensity
        }

        // automatically interpolate your data to the requested mesh
        return interpolate(*mesh, computed_light_intensity, WrappedMesh<2>(destination_mesh, this->geometry),
                           defInterpolation<INTERPOLATION_LINEAR>(interpolation_method));
    }
\endcode

The important objects of the above method are the first and the last lines. In the former one, we check if the computations have been performed
and are up-to-date (remember, we have cleared the value of \c outNeff in \c onInvalidate()). Otherwise we throw an exception. In the last line
we use plask::interpolate function to interpolate our data to the receiver mesh (which is provided as \c destination_mesh argument).

Helper class WrappedMesh helps to automatically consider mirror and periodic boundaries, so the requested points will be wrapped into
your computational domain correctly. And defInterpolation changes DEFAULT_INTERPOLATION method to some real one.

Our solver can perform computations now. However, if it has any configuration to load, we can read it from XML file. To do this, we should
reimplement \c loadConfiguration method. It reads the configuration from the current XML file using plask::XMLReader, by walking through
the consecutive tags. It is important to call \c parseStandardConfiguration if you encounter unknown tag. Below you have an example:

\code
    void loadConfiguration(XMLReader& reader, Manager& manager) {
        while (reader.requireTagOrEnd()) {
            if (reader.getNodeName() == "newton") {
                newton.tolx = reader.getAttribute<double>("tolx", newton.tolx);
                newton.tolf = reader.getAttribute<double>("tolf", newton.tolf);
                newton.maxstep = reader.getAttribute<double>("maxstep", newton.maxstep);
                reader.requireTagEnd();
            } else if (reader.getNodeName() == "wavelength") {
                std::string = reader.requireTextUntilEnd();
                inWavelength.setValue(boost::lexical_cast<double>(wavelength));
            } else if (reader.getNodeName() == "boundary") {
                manager.readBoundaryConditions(source, boundaryConditionsOnField);
                reader.requireTagEnd();
            } else
                parseStandardConfiguration(reader, manager, "<geometry>, <mesh>, <newton>, or <wavelength>");
        }
    }
\endcode

In the above example we assume that we have some local \p struct with parameters of Newton algorithm: \c tolx, \c tolf, and \c maxstep.
They are read from the corresponding attributes of a &lt;newton&gt; tag, with default value equal to the current value of the corresponding
parameter. Furthermore, user can optionally specify a wavelength, which we set as a specified input value of the inWavelength receiver
(single-value receivers can be connected to providers, however they can also have assigned value as normal variables, although in any case
to read their values you must remember about using parenthesis, e.g. <tt>w&nbsp;=&nbsp;inWavelength()</tt>).

The XML file to read by the above method can look as follows (although you should rather use XML attributes to set simple parameters,
in order to make the XML file consistent for all the solvers).:

\verbatim
<optical lib="finite_diff" solver="FiniteDifferencesCartesian2D">
    <newton tolx="0.0001" tolf="1e-9" maxstep="500" />
    <wavelenght>1000</wavelength>
</optical>
\endverbatim

You can now finish your class definition:

\code
}; // class FiniteDifferencesSolver

}}} // namespace plask::solvers::optical_finite_differences

#endif // PLASK__SOLVERS__OPTICAL_FINITE_DIFFERENCES_H
\endcode


\subsubsection solver_python_interface_tutorial Writing Python interface

Once you have written all the C++ code of your solver, you should export it to the Python interface. To do this, create a subdirectory
named \a python in your solver tree (i.e. \a solvers/optical/finite_diff/python) and copy \a your_solver.cpp from \a solvers/skel/python
there (changing its name e.g. to \a finite_differences_python.cpp).

Then edit the file to export all necessary methods and public fields. The contents of this file should look more less like this:

\code
#include <cmath>
#include <plask/python.hpp>
using namespace plask;
using namespace plask::python;

#include "../finite_differences.h"
using namespace plask::solvers::optical_finite_differences; // use your solver pricate namespace

BOOST_PYTHON_MODULE(fd)
{
    {CLASS(FiniteDifferencesSolver, "FiniteDifferencesCartesian2D",
        "Calculate optical modes and optical field distribution using the finite\n"
        "differences method in Cartesian two-dimensional space.")

     RECEIVER(inWavelength, "Wavelength of the light");

     RECEIVER(inTemperature, "Temperature distribution in the structure");

     PROVIDER(outNeff, "Effective index of the last computed mode");

     PROVIDER(outIntensity, "Light intensity of the last computed mode");

     BOUNDARY_CONDITIONS(boundary, boundaryConditionsOnField, "Boundary conditions of the first kind (constant field)");

     METHOD(compute, compute, "Perform the computations");
    }
}
\endcode

\c BOOST_PYTHON_MODULE macro takes the name of the package with your solver (without quotation marks). Your solver will be accessible to
Python if the user imports this package as:

\code{.py}
import plask.optical.fd
\endcode

The arguments of the \c CLASS macro are your solver class, the name in which it will appear in Python, and short solver documentation
(mind the braces outside of the \c CLASS: they are important if you want to put more than one solver in a single interface file, so they
will appear in a single package).

Next you define your exported class member fields, properties (fake fields, which call your class methods on reading or assignment), methods,
providers, and receivers (you have probably noticed that providers and receivers are just class member fields, but they need to be exported
using separate macros, due to some additional logic necessary). Below, there is a complete list of macros exporting class objects and we believe
it is self-explanatory:

\code
METHOD(python_method_name, method_name, "Short documentation", "name_or_argument_1", arg("name_of_argument_2")=default_value_of_arg_2, ...);

RO_FIELD(field_name, "Short documentation");                                                    // read-only field

RW_FIELD(field_name, "Short documentation");                                                    // read-write field

RO_PROPERTY(python_property_name, get_method_name, "Short documentation");                      // read-only property

RW_PROPERTY(python_property_name, get_method_name, set_method_name, "Short documentation");     // read-write property

RECEIVER(inReceiver, "Short documentation");                                                    // receiver in the solver

PROVIDER(outProvider, "Short documentation");                                                   // provider in the solver
\endcode

When defining methods that take some arguments, you need to put their names after the documentation, so the user can set
them by name, as for typical Python methods. You can also specify some default values of some arguments as shown above.

After you successfully compile the solver and the Python interface (just run \c cmake and \c make), the user should be able to execute
the following Python code:

\code{.py}
import numpy
import plask
import plask.optical.fd

solver = plask.optical.fd.FiniteDifferencesCartesian2D()
solver.geometry = plask.geometry.Geometry2DCartesian(plask.geometry.Rectangle(2, 1, "GaN"))
solver.mesh = plask.mesh.RectilinearMesh2D(numpy.linspace(0,2), numpy.linspace(0,1))
solver.inTemperature = 280
solver.compute()
print(solver.outNeff())
\endcode

\subsubsection solver_testing_tutorial Writing automatic solver tests

TODO


This concludes our short tutorial. Now you can go on and write your own calculation solver. Good luck!


*/

#include <string>

#include "log/log.h"
#include "log/data.h"
#include "mesh/mesh.h"
#include "geometry/space.h"
#include "geometry/reader.h"
#include "manager.h"

namespace plask {

struct Manager;

/**
 * Base class for all solvers.
 *
 * @see @ref solvers
 */
class Solver {

    /// Id of the instance of this solver
    std::string solver_name;

  protected:

    /// @c true only if solver is initialized
    bool initialized;

    /**
     * Initialize the solver.
     *
     * Default implementation just does nothing, however it is a good idea to overwrite it in subclasses and put initialization code in it.
     */
    virtual void onInitialize() {}

    /**
     * This method is called by invalidate() to reset stored values.
     *
     * Default implementation does nothing.
     * @see invalidate()
     */
    virtual void onInvalidate() {}

    /**
     * This should be called on beginning of each calculation method to ensure that solver will be initialized.
     * It's does nothing if solver is already initialized and calls onInitialize() if it's not.
     * @return @c true only if solver was already initialized (before calling initCalculation)
     */
    bool initCalculation();

  public:

    Solver(const std::string& name=""): solver_name(name), initialized(false) {}

    /// Virtual destructor (for subclassing). Do nothing.
    virtual ~Solver() {}

    /**
     * Load configuration from given @p source.
     *
     * XML reader (@p source) point to opening of @c this solver tag and
     * after return from this method should point to this solver closing tag.
     *
     * @param source source of configuration
     * @param manager manager from which information about geometry, meshes, materials, and so on can be get if needed
     */
    virtual void loadConfiguration(XMLReader& source, Manager& manager);

    /**
     * Load standard configuration (geometry, mesh) tags from \p source.
     * Throws an exception if the current tag is not a standard tag (so parse your own solver configuration first).
     *
     * \param[in, out] source source of configuration which point to opening of parameter tag
     * \param[in, out] manager manager from which information about geometry, meshes, materials, and so on can be get if needed
     * \param[in] expected_msg optional message stating what was expected
     *
     * \throw XMLUnexpectedElementException if the current tag is not recognized
     */
    void parseStandardConfiguration(XMLReader& source, Manager& manager, const std::string& expected_msg="solver configuration element");

    /**
     * Check if solver is already initialized.
     * @return @c true only if solver is already initialized
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
            writelog(LOG_INFO, "Invalidating solver");
            onInvalidate();
        }
    }

    /**
     * Get name of solver.
     * @return name of this solver
     */
    virtual std::string getClassName() const = 0;

    /**
     * Get solver id
     * \return id of this solver
     */
    inline std::string getId() const {
        std::string name = "";
        if (solver_name != "") {
            name += solver_name; name += ":";
        }
        return name + getClassName();
    }

    /// \return solver name
    inline std::string getName() const { return solver_name; }

    /**
     * Get a description of this solver.
     * @return description of this solver (empty string by default)
     */
    virtual std::string getClassDescription() const { return ""; }

    template<typename ArgT = double, typename ValT = double>
    Data2DLog<ArgT, ValT> dataLog(const std::string& chart_name, const std::string& axis_arg_name, const std::string& axis_val_name) {
        return Data2DLog<ArgT, ValT>(getId(), chart_name, axis_arg_name, axis_val_name);
    }


    template<typename ArgT = double, typename ValT = double>
    Data2DLog<ArgT, ValT> dataLog(const std::string& axis_arg_name, const std::string& axis_val_name) {
        return Data2DLog<ArgT, ValT>(getId(), axis_arg_name, axis_val_name);
    }

    /**
    * Log a message for this solver
    * \param level log level to log
    * \param msg log message
    * \param params parameters passed to format
    **/
    template<typename ...Args>
    void writelog(LogLevel level, std::string msg, Args&&... params) const { plask::writelog(level, getId() + ": " + msg, std::forward<Args>(params)...); }

    //virtual shared_ptr<const Geometry> getUsedGeometry() const;   //return empty by default
    //virtual shared_ptr<const Mesh> getUsedMesh() const;

};

/*
 * Function constructing solver on library loadConfiguration
 */
#define SOLVER_CONSTRUCT_FUNCTION_SUFFIX "_solver_factory"
extern "C" typedef Solver* solver_construct_f(const std::string& name);


template <typename, typename> class SolverWithMesh;

/**
 * Base class for all solvers operating on specified space
 */
template <typename SpaceT>
class SolverOver: public Solver {

    void diconnectGeometry() {
        if (this->geometry)
            this->geometry->changedDisconnectMethod(this, &SolverOver<SpaceT>::onGeometryChangeInternal);
    }

    void onGeometryChangeInternal(const Geometry::Event& evt) {
        this->onGeometryChange(evt);
        this->regenerateMesh();
    }

    // By default this class does not have the mesh, so do nothing
    virtual void regenerateMesh() {}

    template <typename, typename> friend class SolverWithMesh;

  protected:

    /// Space in which the calculations are performed
    shared_ptr<SpaceT> geometry;

    /**
     * Read boundary conditions using information about the geometry of this solver
     * \param manager manager used for load
     * \param reader current XML reader
     * \param dest BoundaryConditions variable which will store the boundary conditions
     */
    template <typename MeshT, typename ConditionT>
    void readBoundaryConditions(Manager& manager, XMLReader& reader, BoundaryConditions<MeshT, ConditionT>& dest) {
        manager.readBoundaryConditions<MeshT, ConditionT>(reader, dest, geometry);
    }

  public:

    /// of the space for this solver
    typedef SpaceT SpaceType;

    SolverOver(const std::string& name="") : Solver(name) {}

    ~SolverOver() {
        diconnectGeometry();
    }

    virtual void loadConfiguration(XMLReader& source, Manager& manager);

    void parseStandardConfiguration(XMLReader& source, Manager& manager, const std::string& expected_msg="solver configuration element");

    /**
     * This method is called when calculation space (geometry) was changed.
     * It just calls invalidate(); but subclasses can customize it.
     * @param evt information about geometry changes
     */
    virtual void onGeometryChange(const Geometry::Event& evt) {
        this->invalidate();
    }


    /**
     * Get current solver geometry space.
     * @return current solver geometry space
     */
    inline shared_ptr<SpaceT> getGeometry() const { return geometry; }

    /**
     * Set new geometry for the solver
     * @param geometry new geometry space
     */
    void setGeometry(const shared_ptr<SpaceT>& geometry) {
        if (geometry == this->geometry) return;
        writelog(LOG_INFO, "Attaching geometry to the solver");
        diconnectGeometry();
        this->geometry = geometry;
        if (this->geometry)
            this->geometry->changedConnectMethod(this, &SolverOver<SpaceT>::onGeometryChangeInternal);
        onGeometryChangeInternal(Geometry::Event(*geometry, 0));
    }
};

/**
 * Base class for all solvers operating on specified olding an external mesh
 */
template <typename SpaceT, typename MeshT>
class SolverWithMesh: public SolverOver<SpaceT> {

    shared_ptr<MeshGeneratorOf<MeshT>> mesh_generator;

    void diconnectMesh() {
        if (this->mesh)
            this->mesh->changedDisconnectMethod(this, &SolverWithMesh<SpaceT, MeshT>::onMeshChange);
    }

    virtual void regenerateMesh() {
        if (this->mesh_generator && this->geometry) {
            auto gen = mesh_generator; // setMesh will reset generator
            setMesh((*mesh_generator)(this->geometry->getChild()));
            mesh_generator = gen;
        }
    }

  protected:

    /// Mesh over which the calculations are performed
    shared_ptr<MeshT> mesh;

  public:

    /// Type of the mesh for this solver
    typedef MeshT MeshType;

    SolverWithMesh(const std::string& name="") : SolverOver<SpaceT>(name) {}

    ~SolverWithMesh() {
        diconnectMesh();
    }

    virtual void loadConfiguration(XMLReader& source, Manager& manager);

    void parseStandardConfiguration(XMLReader& source, Manager& manager, const std::string& expected_msg="solver configuration element");

    /**
     * This method is called when mesh was changed.
     * It just calls invalidate(); but subclasses can customize it.
     * @param evt information about mesh changes
     */
    virtual void onMeshChange(const typename MeshT::Event& evt) {
        this->invalidate();
    }

	/**
     * Get current module mesh.
     *
     * It doesn't check if mesh is non-null.
     * @return current module mesh, dereferenced
     */
    inline MeshT& meshRef() const { return *mesh; }

    /**
     * Get current solver mesh.
     * @return current solver mesh
     */
    inline shared_ptr<MeshT> getMesh() const { return mesh; }

    /**
     * Set new mesh for the solver
     * @param mesh new mesh
     */
    void setMesh(const shared_ptr<MeshT>& mesh) {
        mesh_generator.reset();
        if (mesh == this->mesh) return;
        this->writelog(LOG_INFO, "Attaching mesh to the solver");
        diconnectMesh();
        this->mesh = mesh;
        if (this->mesh)
            this->mesh->changedConnectMethod(this, &SolverWithMesh<SpaceT, MeshT>::onMeshChange);
        typename MeshT::Event event (*mesh, 0);
        onMeshChange(event);
    }

    /**
     * Set new mesh got from generator
     * \param generator mesh generator
     */
    void setMesh(const shared_ptr<MeshGeneratorOf<MeshT>>& generator) {
        mesh_generator = generator;
        regenerateMesh();
    }
};

}       //namespace plask

#include "manager.h" // Just in case module author includes only "solver.h"

namespace plask {

template <typename SpaceT>
void SolverOver<SpaceT>::loadConfiguration(XMLReader& reader, Manager& manager) {
        while (reader.requireTagOrEnd()) parseStandardConfiguration(reader, manager, "<geometry>");
}

template <typename SpaceT, typename MeshT>
void SolverWithMesh<SpaceT, MeshT>::loadConfiguration(XMLReader& reader, Manager& manager) {
        while (reader.requireTagOrEnd()) parseStandardConfiguration(reader, manager, "<geometry> or <mesh>");
}


template <typename SpaceT>
void SolverOver<SpaceT>::parseStandardConfiguration(XMLReader& reader, Manager& manager, const std::string& expected_msg) {
    if (reader.getNodeName() == "geometry") {
        auto name = reader.getAttribute("ref");
        if (!name) name.reset(reader.requireTextInCurrentTag());
        else reader.requireTagEnd();
        auto found = manager.geometries.find(*name);
        if (found == manager.geometries.end())
            throw BadInput(this->getId(), "Geometry '%1%' not found.", *name);
        else {
            auto geometry = dynamic_pointer_cast<SpaceT>(found->second);
            if (!geometry) throw BadInput(this->getId(), "Geometry '%1%' of wrong type.", *name);
            this->setGeometry(geometry);
        }
    } else {
        Solver::parseStandardConfiguration(reader, manager, expected_msg);
    }
}

template <typename SpaceT, typename MeshT>
void SolverWithMesh<SpaceT, MeshT>::parseStandardConfiguration(XMLReader& reader, Manager& manager, const std::string& expected_msg) {
    if (reader.getNodeName() == "mesh") {
        auto name = reader.getAttribute("ref");
        if (!name) name.reset(reader.requireTextInCurrentTag());
        else reader.requireTagEnd();
        auto found = manager.meshes.find(*name);
        if (found != manager.meshes.end()) {
            auto mesh = dynamic_pointer_cast<MeshT>(found->second);
            if (!mesh) throw BadInput(this->getId(), "Mesh '%1%' of wrong type.", *name);
            this->setMesh(mesh);
        }
        else {
            auto found = manager.generators.find(*name);
            if (found != manager.generators.end()) {
                auto generator = dynamic_pointer_cast<MeshGeneratorOf<MeshT>>(found->second);
                if (!generator) throw BadInput(this->getId(), "Mesh '%1%' of wrong type.", *name);
                this->setMesh(generator);
            } else
                throw BadInput(this->getId(), "Neither mesh nor mesh generator '%1%' found.", *name);
        }
    } else {
        SolverOver<SpaceT>::parseStandardConfiguration(reader, manager, expected_msg);
    }
}

}   // namespace plask

#endif // PLASK__SOLVER_H
