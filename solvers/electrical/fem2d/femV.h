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
template<typename Geometry2Dtype> struct FiniteElementMethodElectrical2DSolver: public SolverWithMesh<Geometry2Dtype, RectilinearMesh2D>
{
protected:

    /// Main Matrix
    double **mpA;
    int mAWidth, mAHeight;
    std::string mVChange; // absolute or relative
    int mLoopLim; // number of loops - stops the calculations
    double mVCorrLim; // small-enough correction - stops the calculations
    double mBigNum; // for the first boundary condtion (see: set Matrix)
    double mJs; // p-n junction parameter [A/m^2]
    double mBeta; // p-n junction parameter [1/V]
    double mCondJuncX0; // initial electrical conductivity for p-n junction in x-direction [1/(Ohm*m)]
    double mCondJuncY0; // initial electrical conductivity for p-n junction in y-direction [1/(Ohm*m)]
    double mCondPcontact; // p-contact electrical conductivity [S/m]
    double mCondNcontact; // n-contact electrical conductivity [S/m]
    bool mLogs; // logs (0-most important logs, 1-all logs)
    int mLoopNo; // number of completed loops
    double mMaxAbsVCorr; // max. absolute potential correction
    double mMaxRelVCorr; // max. relative potential correction
    double mMaxVCorr; // max. absolute potential correction (useful for calculations with internal loops)

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
    void savePotentials(); // [V]

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
    BoundaryConditions<RectilinearMesh2D,double> mVConst;

    typename ProviderFor<Potential, Geometry2Dtype>::Delegate outPotential;

    typename ProviderFor<CurrentDensity2D, Geometry2Dtype>::Delegate outCurrentDensity;

    typename ProviderFor<HeatDensity, Geometry2Dtype>::Delegate outHeatDensity;

    ReceiverFor<Temperature, Geometry2Dtype> inTemperature;

    ReceiverFor<Wavelength> inWavelength; // wavelength (for heat generation in the active region) [nm]

    DataVector<const double> getPotentials(const MeshD<2>& dst_mesh, InterpolationMethod method) const;

    DataVector<const double> getHeatDensities(const MeshD<2>& dst_mesh, InterpolationMethod method) const;

    DataVector<const Vec<2> > getCurrentDensities(const MeshD<2>& dst_mesh, InterpolationMethod method) const;

    /**
     * Run potential calculations
     * \return max correction of potential agains the last call
     **/
    double runCalc(int iLoopLim=0);

    /**
     * Run single potential calculations
     * \return max correction of potential agains the last call
     **/
    double runSingleCalc();

    /**
     * Get max absolute correction for potential
     * \return get max absolute correction for potential
     **/
    double getMaxAbsVCorr(); // result in [V]

    /**
     * Get max relative correction for potential
     * \return get max relative correction for potential
     **/
    double getMaxRelVCorr(); // result in [%]

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

}}} // namespaces

#endif

