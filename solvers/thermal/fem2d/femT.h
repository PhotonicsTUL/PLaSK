/**
 * \file
 * Sample solver header for your solver
 */
#ifndef PLASK__MODULE_THERMAL_FINITET_H
#define PLASK__MODULE_THERMAL_FINITET_H

#include <plask/plask.hpp>
#include "node2D.h"
#include "element2D.h"

namespace plask { namespace solvers { namespace thermal {

/// Boundary condition: convection
struct Convection
{
    double mConvCoeff; ///< convection coefficient [W/(m^2*K)]
    double mTAmb1; ///< ambient temperature [K]
    Convection(double coeff, double amb): mConvCoeff(coeff), mTAmb1(amb) {}
    Convection() = default;
};

/// Boundary condition: radiation
struct Radiation
{
    double mSurfEmiss; ///< surface emissivity [-]
    double mTAmb2; ///< ambient temperature [K]
    Radiation(double emiss, double amb): mSurfEmiss(emiss), mTAmb2(amb) {}
    Radiation() = default;
};

/**
 * Solver performing calculations in 2D Cartesian or Cylindrical space using finite element method
 */
template<typename Geometry2Dtype> struct FiniteElementMethodThermal2DSolver: public SolverWithMesh<Geometry2Dtype, RectilinearMesh2D>
{
protected:

    /// Main Matrix
    double **mpA;
    int mAWidth, mAHeight;
    std::string mTChange; // absolute or relative
    int mLoopLim; // number of loops - stops the calculations
    double mTCorrLim; // small-enough correction - stops the calculations
    double mTBigCorr; // big-enough correction for the temperature
    double mBigNum; // for the first boundary condtion (see: set Matrix)
    double mTInit; // ambient temperature
    bool mLogs; // logs (0-most important logs, 1-all logs)
    int mLoopNo; // number of completed loops
    double mMaxAbsTCorr; // max. absolute temperature correction (useful for single calculations managed by external python script)
    double mMaxRelTCorr; // max. relative temperature correction (useful for single calculations managed by external python script)
    double mMaxTCorr; // max. absolute temperature correction (useful for calculations with internal loops)

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
    void saveTemperatures(); // [K]

    /// Create 2D-vector with calculated heat fluxes
    void saveHeatFluxes(); // [W/m^2]

    /// Show vector with calculated temperatures
    void showTemperatures();

    /// Show vector with calculated heat fluxes
    void showHeatFluxes();

    /// Matrix solver
    int solveMatrix(double **ipA, long iN, long iBandWidth);

    /// Do some steps which are the same both for loop- and single- calculations
    void doSomeSteps();

    /// Do more steps which are the same both for loop- and single- calculations
    void doMoreSteps();

    /// Initialize the solver
    virtual void onInitialize();

    /// Invalidate the data
    virtual void onInvalidate();

  public:

    /// Boundary conditions
    BoundaryConditions<RectilinearMesh2D,double> mTConst; // constant temperature [K]
    BoundaryConditions<RectilinearMesh2D,double> mHFConst; // constant heat flux [W/m^2]
    BoundaryConditions<RectilinearMesh2D,Convection> mConvection; // convection
    BoundaryConditions<RectilinearMesh2D,Radiation> mRadiation; // radiation

    typename ProviderFor<Temperature, Geometry2Dtype>::Delegate outTemperature;

    typename ProviderFor<HeatFlux2D, Geometry2Dtype>::Delegate outHeatFlux;

    ReceiverFor<HeatDensity, Geometry2Dtype> inHeatDensity;

    DataVector<const double> getTemperatures(const MeshD<2>& dst_mesh, InterpolationMethod method) const;

    DataVector<const Vec<2> > getHeatFluxes(const MeshD<2>& dst_mesh, InterpolationMethod method) const;

    /**
     * Run temperature calculations
     * \return max correction of temperature agains the last call
     **/
    double runCalc();

    /**
     * Run single temperature calculations
     * \return max correction of temperature agains the last call
     **/
    double runSingleCalc();

    /**
     * Get max absolute correction for temperature
     * \return get max absolute correction for temperature
     **/
    double getMaxAbsTCorr(); // result in [K]

    /**
     * Get max relative correction for temperature
     * \return get max relative correction for temperature
     **/
    double getMaxRelTCorr(); // result in [%]

    void setLoopLim(int iLoopLim);
    void setTCorrLim(double iTCorrLim);
    void setTBigCorr(double iTBigCorr);
    void setBigNum(double iBigNum);
    void setTInit(double iTInit);

    int getLoopLim();
    double getTCorrLim();
    double getTBigCorr();
    double getBigNum();
    double getTInit();

    virtual void loadConfiguration(XMLReader& source, Manager& manager); // for solver configuration (see: *.xpl file with structures)

    FiniteElementMethodThermal2DSolver(const std::string& name="");

    virtual std::string getClassName() const;

    ~FiniteElementMethodThermal2DSolver();
};

}} //namespaces

template <> inline solvers::thermal::Convection parseCondition<solvers::thermal::Convection>(const XMLReader& tag_with_value)
{
    return solvers::thermal::Convection(tag_with_value.requireAttribute<double>("coefficient"), tag_with_value.requireAttribute<double>("Tamb"));
}

template <> inline solvers::thermal::Radiation parseCondition<solvers::thermal::Radiation>(const XMLReader& tag_with_value)
{
    return solvers::thermal::Radiation(tag_with_value.requireAttribute<double>("emissivity"), tag_with_value.requireAttribute<double>("Tamb"));
}

} // namespace plask

#endif

