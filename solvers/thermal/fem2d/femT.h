/**
 * \file
 * Sample solver header for your solver
 */
#ifndef PLASK__MODULE_THERMAL_FINITET_H
#define PLASK__MODULE_THERMAL_FINITET_H

#include <plask/plask.hpp>
#include "node2D.h"
#include "element2D.h"
#include "constants.h"

namespace plask { namespace solvers { namespace thermal {

/**
 * Solver performing calculations in 2D Cartesian space using finite element method
 */
struct FiniteElementMethodThermalCartesian2DSolver: public SolverWithMesh<Geometry2DCartesian, RectilinearMesh2D> {
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
    std::vector<double> mTCorr;
    int mLoopLim; // number of loops - stops the calculations
    double mTCorrLim; // small-enough correction - stops the calculations
    double mTBigCorr; // big-enough correction for the temperature
    double mBigNum; // for the first boundary condtion (see: set Matrix)
    double mTAmb; // ambient temperature
    bool mLogs; // logs (0-most important logs, 1-all logs)
    int mLoopNo; // number of completed loops

    DataVector<double> mTemperatures; // out
    DataVector<Vec<2> > mHeatFluxes; // out
    DataVector<const double> mHeatDensities; // in

    /// Vector of nodes
    std::vector<Node2D> mNodes;

    /// Vector of elements
    std::vector<Element2D> mElements;

    /// Set nodes
    void setNodes();

    /// Set elements
    void setElements();

    /// Set heat densities
    void setHeatDensities();

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

    /// Create vector with calculated temperatures
    void saveTemperatures();

    /// Create 2D-vector with calculated heat fluxes
    void saveHeatFluxes();

    /// Show vector with calculated temperatures (node numbers for info only)
    void showTemperatures();

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
    BoundaryConditions<RectilinearMesh2D,double> mTConst;

    ProviderFor<Temperature, Geometry2DCartesian>::Delegate outTemperature;

    ProviderFor<HeatFlux2D, Geometry2DCartesian>::Delegate outHeatFlux;

    ReceiverFor<HeatDensity, Geometry2DCartesian> inHeatDensity;

    DataVector<const double> getTemperatures(const MeshD<2>& dst_mesh, InterpolationMethod method) const;

    DataVector<const Vec<2> > getHeatFluxes(const MeshD<2>& dst_mesh, InterpolationMethod method) const;

    /// Run temperature calculations
    void runCalc();

    void setLoopLim(int iLoopLim);
    void setTCorrLim(double iTCorrLim);
    void setTBigCorr(double iTBigCorr);
    void setBigNum(double iBigNum);
    void setTAmb(double iTAmb);

    int getLoopLim();
    double getTCorrLim();
    double getTBigCorr();
    double getBigNum();
    double getTAmb();

    virtual void loadParam(const std::string& param, XMLReader& source, Manager& manager); // for solver configuration (see: *.xpl file with structures)

    FiniteElementMethodThermalCartesian2DSolver(const std::string& name="");

    virtual std::string getClassName() const { return "CartesianFEM"; }

    ~FiniteElementMethodThermalCartesian2DSolver();
};


}}} // namespace

#endif

