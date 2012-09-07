/**
 * \file
 * Sample solver header for your solver
 */
#ifndef PLASK__MODULE_YOUR_MODULE_H
#define PLASK__MODULE_YOUR_MODULE_H

#include <plask/plask.hpp>

namespace plask { namespace solvers { namespace gain_trivial {

/**
 * This is Doxygen documentation of your solver.
 * Write a brief description of it.
 */
template <typename GeometryT>
class StepProfileGain: public SolverOver<GeometryT> {

    /// Element for which we specify the gain
    weak_ptr<const GeometryElementD<GeometryT::DIMS>> element;

    ///
    PathHints hints;

  public:

    /// Gain value to return
    double gain;

    /// Sample provider for field (it's better to use delegate here).
    typename ProviderFor<Gain, GeometryT>::Delegate outGain;


    StepProfileGain(const std::string& name=""): SolverOver<GeometryT>(name)/*, outGain(this, &StepProfileGain<GeometryT>::getGain)*/ {}

    virtual std::string getClassName() const;

    virtual std::string getClassDescription() const {
        return "It returns a giveen step-profile gain on a specified element";
    }

    virtual void loadParam(const std::string& param, XMLReader& reader, Manager& manager);

    /**
     * Set element, on which there is a gain.
     * \param element the geometry element
     * \param path optional path hints specifying the element
     */
    void setElement(const weak_ptr<const GeometryElementD<GeometryT::DIMS>>& element, const PathHints& path=PathHints());

  protected:

    /// Method computing the value for the delegate provider
    const DataVector<double> getGain(const plask::MeshD<GeometryT::DIMS>& dst_mesh, double wavelength, plask::InterpolationMethod method=DEFAULT_INTERPOLATION);

};


}}} // namespace

#endif

