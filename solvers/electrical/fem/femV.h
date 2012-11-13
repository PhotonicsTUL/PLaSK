#ifndef PLASK__MODULE_ELECTRICAL_FEMV_H
#define PLASK__MODULE_ELECTRICAL_FEMV_H

#include <plask/plask.hpp>

#include "band_matrix.h"

namespace plask { namespace solvers { namespace electrical {

/// Choice of matrix factorization algorithms
enum Algorithm {
    ALGORITHM_SLOW,	///< slow algorithm using BLAS level 2
    ALGORITHM_BLOCK	///< block algorithm (thrice faster, however a little prone to failures)
    //ITERATIVE_ALGORITHM
};

/// Type of the returned correction
enum CorrectionType {
    CORRECTION_ABSOLUTE,    ///< absolute correction is used
    CORRECTION_RELATIVE     ///< relative correction is used
};

/// Choice of heat computation method in active region
enum HeatMethod {
    HEAT_JOULES, ///< compute Joules heat using effective conductivity
    HEAT_BANDGAP ///< compute heat based on the size of the band gap
};


/**
 * Solver performing calculations in 2D Cartesian or Cylindrical space using finite element method
 */
template<typename Geometry2DType>
struct FiniteElementMethodElectrical2DSolver: public SolverWithMesh<Geometry2DType, RectilinearMesh2D> {

  protected:

    double mBigNum;      ///< A big number used for the first boundary condition (see: set Matrix) (TODO: remove this and apply conditions by hand)

    int mAOrder,         ///< Number of columns in the main matrix
        mABand;          ///< Number of non-zero rows on and below the main diagonal of the main matrix

    double mJs;          ///< p-n junction parameter [A/m^2]
    double mBeta;        ///< p-n junction parameter [1/V]
    double mCondJuncX0;  ///< initial electrical conductivity for p-n junction in x-direction [S/m]
    double mCondJuncY0;  ///< initial electrical conductivity for p-n junction in y-direction [S/m]
    double mCondPcontact;///< p-contact electrical conductivity [S/m]
    double mCondNcontact;///< n-contact electrical conductivity [S/m]

    double mVCorrLim;     ///< Maximum voltage correction accepted as convergence
    int mLoopNo;          ///< Number of completed loops
    double mMaxAbsVCorr;  ///< Maximum absolute voltage correction (useful for single calculations managed by external python script)
    double mMaxRelVCorr;  ///< Maximum relative voltage correction (useful for single calculations managed by external python script)
    double mMaxVCorr;     ///< Maximum absolute voltage correction (useful for calculations with internal loops)
    double mDV;           ///< Maximum voltage

    DataVector<std::pair<double,double>> mCond;     ///< Cached element conductivities
    DataVector<double> mPotentials;                 ///< Computed potentials
    DataVector<Vec<2,double>> mCurrentDensities;    ///< Computed current densities
    DataVector<double> mHeatDensities;              ///< Computed and cached heat source densities

    /// Set stiffness matrix + load vector
    void setMatrix(BandSymMatrix& oA, DataVector<double>& oLoad,
                   const BoundaryConditionsWithMesh<RectilinearMesh2D,double>& iVConst
                  );

    /// Save conductivities (excluding active region)
    void saveConductivities();

    /// Update stored potentials and calculate corrections
    void savePotentials(DataVector<double>& iV);

    /// Update stored current densities
    void saveCurrentDensities();

    /// Create 2D-vector with calculated heat densities
    void saveHeatDensities();

    /// Matrix solver
    int solveMatrix(BandSymMatrix& iA, DataVector<double>& ioB);

    /// Initialize the solver
    virtual void onInitialize();

    /// Invalidate the data
    virtual void onInvalidate();

  public:

    CorrectionType mCorrType; ///< Type of the returned correction

    HeatMethod mHeatMethod; ///< Method of heat computation

    /// Boundary condition
    BoundaryConditions<RectilinearMesh2D,double> mVConst;

    typename ProviderFor<Potential, Geometry2DType>::Delegate outPotential;

    typename ProviderFor<CurrentDensity2D, Geometry2DType>::Delegate outCurrentDensity;

    typename ProviderFor<HeatDensity, Geometry2DType>::Delegate outHeatDensity;

    ReceiverFor<Temperature, Geometry2DType> inTemperature;

    ReceiverFor<Wavelength> inWavelength; /// wavelength (for heat generation in the active region) [nm]

    Algorithm mAlgorithm;   ///< Factorization algorithm to use

    /**
     * Run temperature calculations
     * \return max correction of temperature against the last call
     **/
    double compute(int iLoopLim=1);

    /**
     * Get max absolute correction for temperature
     * \return get max absolute correction for temperature
     **/
    double getMaxAbsVCorr() const { return mMaxAbsVCorr; } // result in [K]

    /**
     * Get max relative correction for temperature
     * \return get max relative correction for temperature
     **/
    double getMaxRelVCorr() const { return mMaxRelVCorr; } // result in [%]

    double getVCorrLim() const { return mVCorrLim; }
    void setVCorrLim(double iVCorrLim) { mVCorrLim = iVCorrLim; }

    double getBigNum() { return mBigNum; }
    void setBigNum(double iBigNum)  { mBigNum = iBigNum; }

    double getBeta() const { return mBeta; }
    void setBeta(double iBeta)  { mBeta = iBeta; }

    double getJs() const { return mJs; }
    void setJs(double iJs)  { mJs = iJs; }

    double getCondPcontact() const { return mCondPcontact; }
    void setCondPcontact(double iCondPcontact)  { mCondPcontact = iCondPcontact; }

    double getCondNcontact() const { return mCondNcontact; }
    void setCondNcontact(double iCondNcontact)  { mCondNcontact = iCondNcontact; }

    std::pair<double,double> getCondJunc0() const { return std::make_pair(mCondJuncX0, mCondJuncY0); }
    void setCondJunc0(const std::pair<double,double>& iCond)  { mCondJuncX0 = iCond.first; mCondJuncY0 = iCond.second; }

    virtual void loadConfiguration(XMLReader& source, Manager& manager); // for solver configuration (see: *.xpl file with structures)

    FiniteElementMethodElectrical2DSolver(const std::string& name="");

    virtual std::string getClassName() const;

    ~FiniteElementMethodElectrical2DSolver();

  protected:

    DataVector<const double> getPotentials(const MeshD<2>& dst_mesh, InterpolationMethod method) const;

    DataVector<const double> getHeatDensities(const MeshD<2>& dst_mesh, InterpolationMethod method);

    DataVector<const Vec<2>> getCurrentDensities(const MeshD<2>& dst_mesh, InterpolationMethod method);

};

}} //namespaces

} // namespace plask

#endif

