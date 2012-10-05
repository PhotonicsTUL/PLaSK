/**
 * \file
 * Sample solver header for your solver
 */
#ifndef PLASK__MODULE_ELECTRICAL_FINITEV_H
#define PLASK__MODULE_ELECTRICAL_FINITEV_H

#include <plask/plask.hpp>
#include "node2D.h"
#include "element2D.h"

namespace plask { namespace solvers { namespace electrical {

/**
 * Solver performing calculations in 2D Cartesian or Cylindrical space using finite element method
 */
template<typename Geometry2Dtype> struct FiniteElementMethodElectrical2DSolver: public SolverWithMesh<Geometry2Dtype, RectilinearMesh2D> {
/*
    /// Sample receiver for temperature.
    ReceiverFor<Temperature, Space2DCartesian> inTemperature;

    /// Sample provider for simple value.
    ProviderFor<SomeSingleValueProperty>::WithValue outSingleValue;

    /// Sample provider for field (it's better to use delegate here).
    ProviderFor<SomeOnMeshProperty, Space2DCartesian>::Delegate outSomeField;

    YourSolver():
        outDelegateProvider(this, getDelegated) // getDelegated will be called whether provider value is requested
    {
        inTemperature = 300.; // temperature receiver has some sensible value
    }

    virtual std::string getClassDescription() const {
        return "This solver does this and that. And this description can be e.g. shown as a hng in GUI.";
    }*/

/**
    * This method performs the main computations.
    * Add short description of every method in a comment like this.
    * \param parameter some value to be provided by the user for computationsś
    **/
/*void compute(double parameter) {
        // The code of this method probably should be in cpp file...
        // But below we show some key elements
        initCalculation(); // This must be called before any calculation!
        writelog(LOG_INFO, "Begining calculation of something");
        auto temperature = inTemperature(*mesh); // Obtain temperature from some other solver
        // [...] Do your computations here
        outSingleValue = new_computed_value;
        writelog(LOG_RESULT, "Found new value of something = $1$", new_computed_value);
        outSingleValue.fireChanged(); // Inform other solvers that you have computed a new value
         outSomeField.fireChanged();
    }*/

  protected:

    /// Main Matrix
    double **mpA;
    int mAWidth, mAHeight;
    std::vector<double> mVcorr;
    int mLoopLim; // number of loops - stops the calculations
    double mVCorrLim; // small-enough correction - stops the calculations
    double mVBigCorr; // big-enough correction for the potential
    double mBigNum; // for the first boundary condtion (see: set Matrix)
    double mJs; // p-n junction parameter [A/m^2]
    double mBeta; // p-n junction parameter [1/V]
    double mCondJuncX0; // initial electrical conductivity for p-n junction in x-direction [1/(Ohm*m)]
    double mCondJuncY0; // initial electrical conductivity for p-n junction in y-direction [1/(Ohm*m)]
    bool mLogs; // logs (0-most important logs, 1-all logs)
    int mLoopNo; // number of completed loops

    ReceiverFor<Wavelength> inWavelength; // wavelength (for heat generation in the active region) [nm]

    DataVector<double> mPotentials; // out
    DataVector<Vec<2> > mCurrentDensities; // out
    DataVector<double> mHeatDensities; // out
    DataVector<const double> mTemperatures; // in

    /// Vector of nodes
    std::vector<Node2D> mNodes;

    /// Vector of elements
    std::vector<Element2D> mElements;

    /// Set nodes
    void setNodes();

    /// Set elements
    void setElements();

    /// Set temperatures
    void setTemperatures();

    /// Set matrix
    void setSolver();

    /// Delete vectors
    void delSolver();

    /// Set stiffness matrix + load vector
    void setMatrix();

    /// Update nodes
    void updNodes();

    /// Update elements
    void updElements();

    /// Show info for all nodes
    void showNodes();

    /// Show info for all elements
    void showElements();

    /// Create vector with calculated potentials
    double savePotentials(); // [V]

    /// Create 2D-vector with calculated current densities
    void saveCurrentDensities(); // [A/m^2]

    /// Create vector with calculated heat densities
    void saveHeatDensities(); // [W/m^3]

    /// Show vector with calculated potentials
    void showPotentials();

    /// Show vector with calculated current densities
    void showCurrentDensities();

    /// Show vector with calculated heat fluxes
    void showHeatDensities();

    /// Matrix solver
    int solveMatrix(double **ipA, long iN, long iBandWidth);

    /// Initialize the solver
    virtual void onInitialize();

    /// Invalidate the data
    virtual void onInvalidate();
/*
    /// Method computing the value for the delegate provider
    const DataVector<double> getDelegated(const plask::MeshD<2>& dst_mesh, plask::InterpolationMethod method=DEFAULT_INTERPOLATION) {
        if (!outSingleValue.hasValue())  // this is one possible indication that the solver is invalidated
            throw NoValue(SomeSingleValueProperty::NAME);
        if (method == DEFAULT_INTERPOLATION) method = INTERPOLATION_LINEAR;
        return interpolate(*mesh, my_data, dst_mesh, method); // interpolate your data to the requested mesh
    }*/
  public:

    /// Boundary conditions
    BoundaryConditions<RectilinearMesh2D,double> mVconst;

    typename ProviderFor<Potential, Geometry2Dtype>::Delegate outPotential;

    typename ProviderFor<CurrentDensity2D, Geometry2Dtype>::Delegate outCurrentDensity;

    typename ProviderFor<HeatDensity, Geometry2Dtype>::Delegate outHeatDensity;

    ReceiverFor<Temperature, Geometry2Dtype> inTemperature;

    DataVector<const double> getPotentials(const MeshD<2>& dst_mesh, InterpolationMethod method) const;

    DataVector<const double> getHeatDensities(const MeshD<2>& dst_mesh, InterpolationMethod method) const;

    DataVector<const Vec<2> > getCurrentDensities(const MeshD<2>& dst_mesh, InterpolationMethod method) const;

    /**
     * Run potential calculations
     * \return max correction of temperature agains the last call
     **/
    double runCalc();

    void setLoopLim(int iLoopLim);
    void setVCorrLim(double iVCorrLim);
    void setVBigCorr(double iVBigCorr);
    void setBigNum(double iBigNum);

    int getLoopLim();
    double getVCorrLim();
    double getVBigCorr();
    double getBigNum();

    virtual void loadConfiguration(XMLReader& source, Manager& manager); // for solver configuration (see: *.xpl file with structures)

    FiniteElementMethodElectrical2DSolver(const std::string& name="");

    virtual std::string getClassName() const;

    ~FiniteElementMethodElectrical2DSolver();
};


}}} // namespace

#endif

