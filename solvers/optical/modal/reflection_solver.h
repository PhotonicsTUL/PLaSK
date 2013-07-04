#ifndef PLASK__SOLVER_REFLECTION_SOLVER_H
#define PLASK__SOLVER_REFLECTION_SOLVER_H

#include <plask/plask.hpp>

namespace plask { namespace solvers { namespace modal {

/**
 * Reflection transformation solver in Cartesian 2D geometry.
 */
struct FourierReflection2D: public SolverOver<Geometry2DCartesian> {

    /// Receiver of the wavelength
    ReceiverFor<Wavelength> inWavelength;

    /// Receiver for the temperature
    ReceiverFor<Temperature, Geometry2DCartesian> inTemperature;

    /// Receiver for the gain
    ReceiverFor<Gain, Geometry2DCartesian> inGain;

    /// Provider for computed effective index
    ProviderFor<EffectiveIndex>::WithValue outNeff;

    /// Provider of optical field
    ProviderFor<OpticalIntensity, Geometry2DCartesian>::Delegate outIntensity;




//     /// Sample receiver for temperature.
//     ReceiverFor<Temperature, Geometry2DCartesian> inTemperature;
//
//     /// Sample provider for simple value.
//     ProviderFor<SomeSingleValueProperty>::WithValue outSingleValue;
//
//     /// Sample provider for field (it's better to use delegate here).
//     ProviderFor<SomeFieldProperty, Geometry2DCartesian>::Delegate outSomeField;
//
//     YourSolver(const std::string& name="");
//
//     virtual std::string getClassName() const { return "NameOfYourSolver"; }
//
//     virtual std::string getClassDescription() const {
//         return "This solver does this and that. And this description can be e.g. shown as a hint in GUI.";
//     }
//
//     virtual void loadConfiguration(plask::XMLReader& reader, plask::Manager& manager);
//
//     /**
//      * This method performs the main computations.
//      * Add short description of every method in a comment like this.
//      * \param parameter some value to be provided by the user for computationsś
//      **/
//     void compute(double parameter);
//
//   protected:
//
//     /// This is field data computed by this solver on its own mesh
//     DataVector<double> my_data;
//
//     /// Initialize the solver
//     virtual void onInitialize();
//
//     /// Invalidate the data
//     virtual void onInvalidate();
//
//     /// Method computing the value for the delegate provider
//     const DataVector<const double> getDelegated(const MeshD<2>& dst_mesh, InterpolationMethod method=DEFAULT_INTERPOLATION);
//
    // /*
    //  * Get the position of the matching interface.
    //  *
    //  * \return index of the vertical mesh, where interface is set
    //  */
    // inline size_t getInterface() { return interface; }
    //
    // /*
    //  * Set the position of the matching interface.
    //  *
    //  * \param index index of the vertical mesh, where interface is set
    //  */
    // inline void setInterface(size_t index) {
    //     if (!mesh) setSimpleMesh();
    //     if (index < 0 || index >= mesh->vert().size())
    //         throw BadInput(getId(), "wrong interface position");
    //     log(LOG_DEBUG, "Setting interface at postion %g (mesh index: %d)",  mesh->vert()[index], index);
    //     interface = index;
    // }
    //
    // /*
    //  * Set the position of the matching interface at the top of the provided geometry object
    //  *
    //  * \param path path to the object in the geometry
    //  */
    // void setInterfaceOn(const PathHints& path) {
    //     if (!mesh) setSimpleMesh();
    //     auto boxes = geometry->getLeafsBoundingBoxes(path);
    //     if (boxes.size() != 1) throw NotUniqueObjectException();
    //     interface = std::lower_bound(mesh->vert().begin(), mesh->vert().end(), boxes[0].upper.vert()) - mesh->vert().begin();
    //     if (interface >= mesh->vert().size()) interface = mesh->vert().size() - 1;
    //     log(LOG_DEBUG, "Setting interface at postion %g (mesh index: %d)",  mesh->vert()[interface], interface);
    // }


};


}}} // namespace

#endif

