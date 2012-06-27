/**
 * \file
 * Sample module header for your module
 */
#ifndef PLASK__MODULE_YOUR_MODULE_H
#define PLASK__MODULE_YOUR_MODULE_H

#include <plask/plask.hpp>

namespace plask { namespace modules { namespace your_module {

/**
 * This is Doxygen documentation of your module.
 * Write a brief description of it.
 */
struct YourModule: public ModuleWithMesh<ForExample_Space2DCartesian, ForExample_RectilinearMesh2D> {

    /// Sample receiver for temperature.
    ReceiverFor<Temperature, Space2DCartesian> inTemperature;

    /// Sample provider for simple value.
    ProviderFor<SomeSingleValueProperty>::WithValue outSingleValue;

    /// Sample provider for field (it's better to use delegate here).
    ProviderFor<SomeFieldProperty, Space2DCartesian>::Delegate outSomeField;

    YourModule():
        outDelegateProvider(this, getDelegated) // getDelegated will be called whether provider value is requested
    {
        inTemperature = 300.; // temperature receiver has some sensible value
    }

    virtual std::string getName() const { return "Name of your module"; }

    virtual std::string getDescription() const {
        return "This module does this and that. And this description can be e.g. shown as a hng in GUI.";
    }

    /**
     * This method performs the main computations.
     * Add short description of every method in a comment like this.
     * \param parameter some value to be provided by the user for computationsś
     **/
    void compute(double parameter) {
        // The code of this method probably should be in cpp file...
        // But below we show some key elements
        initCalculation(); // This must be called before any calculation!
        writelog(LOG_INFO, "Begining calculation of something");
        auto temperature = inTemperature(*mesh); // Obtain temperature from some other module
        // [...] Do your computations here
        outSingleValue = new_computed_value;
        writelog(LOG_RESULT, "Found new value of something = $1$", new_computed_value);
        outSingleValue.fireChanged(); // Inform other modules that you have computed a new value
        outSomeField.fireChanged();
    }

  protected:

    /// This is field data computed by this module on its own mesh
    DataVector<double> my_data;

    /// Initialize the module
    virtual void onInitialize() { // In this function check if geometry and mesh are set
        if (!geometry) throw NoGeometryException(getId());
        if (!mesh) throw NoMeshException(getId());
        my_data.reset(mesh->size()); // and e.g. allocate memory
    }

    /// Invalidate the data
    virtual void onInvalidate() { // This will be called when e.g. geometry or mesh changes and your results become outdated
        outSingleValue.invalidate(); // clear the value
        my_data.reset();
        // Make sure that no provider returns any value.
        // If this method has been called, before next computations, onInitialize will be called.
   }

    /// Method computing the value for the delegate provider
    const DataVector<double> getDelegated(const plask::Mesh<2>& dst_mesh, plask::InterpolationMethod method=DEFAULT_INTERPOLATION) {
        if (!outSingleValue.hasValue())  // this is one possible indication that the module is invalidated
            throw NoValue(SomeSingleValueProperty::NAME);
        if (method == DEFAULT_INTERPOLATION) method = INTERPOLATION_LINEAR;
        return interpolate(*mesh, my_data, dst_mesh, method); // interpolate your data to the requested mesh
    }

};


}}} // namespace

#endif

