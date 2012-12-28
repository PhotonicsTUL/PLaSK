/**
 * \file
 * Sample solver header for your solver
 */
#ifndef PLASK__SOLVER_GAIN_FERMI_H
#define PLASK__SOLVER_GAIN_FERMI_H

#include <plask/plask.hpp>

namespace plask { namespace solvers { namespace fermi {

/**
 * Gain solver using Fermi Golden Rule
 */
template <typename GeometryType>
struct FermiGainSolver: public SolverOver<GeometryType>
{
    /**
     * Our own implementation of receivers.
     * As this solver does not have explicit compute function, we must trigger outGain change
     * when any input changes. Receivers of this class do it automatically.
     */
    template <typename Property>
    struct MyReceiverFor: ReceiverFor<Property, GeometryType>
    {
        FermiGainSolver<GeometryType>* parent;

        MyReceiverFor(FermiGainSolver<GeometryType>* parent): parent(parent) {}

        template <typename T> MyReceiverFor& operator=(const T& rhs) {
            ReceiverFor<Property, GeometryType>::operator=(rhs); return *this;
        }

        template <typename T> MyReceiverFor& operator=(T&& rhs) {
            ReceiverFor<Property, GeometryType>::operator=(std::forward<T>(rhs)); return *this;
        }

        virtual void onChange() {
            parent->outGain.fireChanged();  // the input changed, so we inform the world that everybody should get the new gain
        }
    };

    /// Receiver for temperature.
    MyReceiverFor<Temperature> inTemperature;

    /// Receiver for carriers concentration in the active region
    MyReceiverFor<CarriersConcentration> inCarriersConcentration;

    /// Provider for gain distribution
    typename ProviderFor<Gain, GeometryType>::Delegate outGain;

    FermiGainSolver(const std::string& name="");

    virtual std::string getClassName() const;

    virtual void loadConfiguration(plask::XMLReader& reader, plask::Manager& manager);

  protected:

    /// Initialize the solver
    virtual void onInitialize();

    /// Invalidate the gain
    virtual void onInvalidate();

    /**
     * Detect location and materials of quantum wells.
     * Store the results in the class fields
     */
    void detectQuantumWells();

    /**
     * Compute gain at given point. This method is called multiple times when the gain is being provided
     * \param point point to compute gain at
     * \param wavelenght wavelenght to compute gain for
     * \return computed gain
     */
    double computeGain(const Vec<2>& point, double wavelenght);

    /**
     * Method computing the gain on the mesh (called by gain provider)
     * \param dst_mesh destination mesh
     * \param wavelenght wavelenght to compute gain for
     * \return gain distribution
     */
    const DataVector<const double> getGain(const MeshD<2>& dst_mesh, double wavelenght, InterpolationMethod=DEFAULT_INTERPOLATION);

};


}}} // namespace

#endif

