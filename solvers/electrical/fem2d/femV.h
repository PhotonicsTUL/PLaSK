/**
 * \file
 * Sample solver header for your solver
 */
#ifndef PLASK__MODULE_ELECTRICAL_FINITEV_H
#define PLASK__MODULE_ELECTRICAL_FINITEV_H

#include <plask/plask.hpp>
#include "node2D.h"
#include "element2D.h"
#include "constants.h"

namespace plask { namespace solvers { namespace electrical {

/**
 * Solver performing calculations in 2D Cartesian space using finite element method
 */
struct FiniteElementMethodElectricalCartesian2DSolver: public SolverWithMesh<Geometry2DCartesian, RectilinearMesh2D> {
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
    DataVector<double> mPotentials; // out
    DataVector<Vec<2> > mCurrentDensities; // out
    DataVector<double> mHeatDensities; // out
    DataVector<double> mTemperatures; // in

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
    void savePotentials();

    /// Create 2D-vector with calculated current densities
    void saveCurrentDensities();

    /// Create vector with calculated heat densities
    void saveHeatDensities();

    /// Show vector with calculated potentials (node numbers for info only)
    void showPotentials();

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

    ProviderFor<Potential, Geometry2DCartesian>::Delegate outPotential;

    ProviderFor<CurrentDensity2D, Geometry2DCartesian>::Delegate outCurrentDensity;

    ProviderFor<HeatDensity, Geometry2DCartesian>::Delegate outHeatDensity;

    ReceiverFor<Temperature, Geometry2DCartesian> inTemperature;

    DataVector<double> getPotentials(const MeshD<2>& dst_mesh, InterpolationMethod method) const;

    DataVector<double> getHeatDensities(const MeshD<2>& dst_mesh, InterpolationMethod method) const;

    DataVector<Vec<2> > getCurrentDensities(const MeshD<2>& dst_mesh, InterpolationMethod method) const;

    /**
     * Find new potential distribution.
     *
     **/

    /// Run potential calculations
    void runCalc();

    void setLoopLim(int iLoopLim);
    void setVCorrLim(double iVCorrLim);
    void setVBigCorr(double iVBigCorr);
    void setBigNum(double iBigNum);

    int getLoopLim();
    double getVCorrLim();
    double getVBigCorr();
    double getBigNum();

    virtual void loadParam(const std::string& param, XMLReader& source, Manager& manager); // for solver configuration (see: *.xpl file with structures)

    // Parameters for rootdigger
    int mLoopLim; // number of loops - stops the calculations
    double mVCorrLim; // small-enough correction - stops the calculations
    double mVBigCorr; // big-enough correction for the potential
    double mBigNum; // for the first boundary condtion (see: set Matrix)

    FiniteElementMethodElectricalCartesian2DSolver(const std::string& name="");

    virtual std::string getClassName() const { return "CartesianFEM"; }

    ~FiniteElementMethodElectricalCartesian2DSolver();
};


}}} // namespace

#endif

