#ifndef PLASK__SOLVER_SIMPLE_DIFFUSION_CYL_H
#define PLASK__SOLVER_SIMPLE_DIFFUSION_CYL_H

#include <plask/plask.hpp>

namespace plask { namespace solvers { namespace diffusion {

/**
 * This is Doxygen documentation of your solver.
 * Write a brief description of it.
 */
struct SimpleDiffusionSolverCyl: public SolverOver<Geometry2DCylindrical>
{

    /// Receiver for temperature.
    ReceiverFor<Temperature, Geometry2DCylindrical> inTemperature;

    /// Provider for carrier density distribution (it's better to use delegate here).
    ProviderFor<CarrierConcentration, Geometry2DCylindrical>::Delegate outCarrierDensity;

    SimpleDiffusionSolverCyl(const std::string& name="");

    virtual std::string getClassName() const { return "SimpleDiffusionCyl"; }

    virtual std::string getClassDescription() const {
        return "This solver computes carrier distribution for single active region using simple one-dimensional model.";
    }

    virtual void loadConfiguration(XMLReader& source, Manager& manager);


    /**
     * This method performs the main computations.
     * Add short description of every method in a comment like this.
     * \param parameter some value to be provided by the user for computations
     **/
    void compute(double parameter);

  protected:

    /**
     * Detect boxes in which threre are quantum wells.
     *
     * This method assumes that geometry is present. If checks if there is only one column of quantum wells and
     * if they are all of equal sizes. For this simple solver, barriers are ignored.
     *
     * \return list of boxes with quantum wells
     */
    std::deque<Box2D> detectQuantumWells();

    /// Initialize the solver
    virtual void onInitialize();

    /// Invalidate the data
    virtual void onInvalidate();

    /// Method computing the value for the delegate provider
    const DataVector<const double> getCarrierDensity(const MeshD<2>& dst_mesh, InterpolationMethod method=DEFAULT_INTERPOLATION);

};


}}} // namespace

#endif

