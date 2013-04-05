#ifndef PLASK__MODULE_THERMAL_FEMT_H
#define PLASK__MODULE_THERMAL_FEMT_H

#include <plask/plask.hpp>

#include "band_matrix.h"

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


/// Choice of matrix factorization algorithms
enum Algorithm {
    ALGORITHM_SLOW,     ///< slow algorithm using BLAS level 2
    ALGORITHM_CHOLESKY,    ///< block algorithm (thrice faster, however a little prone to failures)
};

/// Type of the returned correction
enum CorrectionType {
    CORRECTION_ABSOLUTE,    ///< absolute correction is used
    CORRECTION_RELATIVE     ///< relative correction is used
};


/**
 * Solver performing calculations in 2D Cartesian or Cylindrical space using finite element method
 */
template<typename Geometry2DType>
struct FiniteElementMethodThermal2DSolver: public SolverWithMesh<Geometry2DType, RectilinearMesh2D> {

  protected:

    int mAsize,         ///< Number of columns in the main matrix
        mAband;          ///< Number of non-zero rows on and below the main diagonal of the main matrix


    double mTCorrLim;     ///< Maximum temperature correction accepted as convergence
    double mTInit;        ///< Initial temperature
    int mLoopNo;          ///< Number of completed loops
    double mMaxT;         ///< Maximum temperature recorded
    double mMaxAbsTCorr;  ///< Maximum absolute temperature correction (useful for single calculations managed by external python script)
    double mMaxRelTCorr;  ///< Maximum relative temperature correction (useful for single calculations managed by external python script)
    double mMaxTCorr;     ///< Maximum absolute temperature correction (useful for calculations with internal loops)

    DataVector<double> mTemperatures;           ///< Computed temperatures

    DataVector<Vec<2,double>> mHeatFluxes;      ///< Computed (only when needed) heat fluxes on our own mesh

    /// Set stiffness matrix + load vector
    void setMatrix(DpbMatrix& oA, DataVector<double>& oLoad,
                   const BoundaryConditionsWithMesh<RectilinearMesh2D,double>& iTConst,
                   const BoundaryConditionsWithMesh<RectilinearMesh2D,double>& iHFConst,
                   const BoundaryConditionsWithMesh<RectilinearMesh2D,Convection>& iConvection,
                   const BoundaryConditionsWithMesh<RectilinearMesh2D,Radiation>& iRadiation
                  );

    /// Update stored temperatures and calculate corrections
    void saveTemperatures(DataVector<double>& iT);

    /// Create 2D-vector with calculated heat fluxes
    void saveHeatFluxes(); // [W/m^2]

    /// Matrix solver
    void solveMatrix(DpbMatrix& iA, DataVector<double>& ioB);

    /// Initialize the solver
    virtual void onInitialize();

    /// Invalidate the data
    virtual void onInvalidate();

  public:

    CorrectionType mCorrType; ///< Type of the returned correction

    // Boundary conditions
    BoundaryConditions<RectilinearMesh2D,double> mTConst; ///< Boundary condition of constant temperature [K]
    BoundaryConditions<RectilinearMesh2D,double> mHFConst; ///< Boundary condition of constant heat flux [W/m^2]
    BoundaryConditions<RectilinearMesh2D,Convection> mConvection; ///< Boundary condition of convection
    BoundaryConditions<RectilinearMesh2D,Radiation> mRadiation; ///< Boundary condition of radiation

    typename ProviderFor<Temperature, Geometry2DType>::Delegate outTemperature;

    typename ProviderFor<HeatFlux2D, Geometry2DType>::Delegate outHeatFlux;

    ReceiverFor<HeatDensity, Geometry2DType> inHeatDensity;

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
    double getMaxAbsTCorr() const { return mMaxAbsTCorr; } // result in [K]

    /**
     * Get max relative correction for temperature
     * \return get max relative correction for temperature
     **/
    double getMaxRelTCorr() const { return mMaxRelTCorr; }// result in [%]

    void setTCorrLim(double iTCorrLim) { mTCorrLim = iTCorrLim; }
    void setTInit(double iTInit)  { mTInit = iTInit; }

    double getTCorrLim() const { return mTCorrLim; }
    double getTInit() const { return mTInit; }

    virtual void loadConfiguration(XMLReader& source, Manager& manager); // for solver configuration (see: *.xpl file with structures)

    FiniteElementMethodThermal2DSolver(const std::string& name="");

    virtual std::string getClassName() const;

    ~FiniteElementMethodThermal2DSolver();

  protected:

    DataVector<const double> getTemperatures(const MeshD<2>& dst_mesh, InterpolationMethod method) const;

    DataVector<const Vec<2> > getHeatFluxes(const MeshD<2>& dst_mesh, InterpolationMethod method);
};

}} //namespaces

template <> inline solvers::thermal::Convection parseBoundaryValue<solvers::thermal::Convection>(const XMLReader& tag_with_value)
{
    return solvers::thermal::Convection(tag_with_value.requireAttribute<double>("coeff"), tag_with_value.requireAttribute<double>("ambient"));
}

template <> inline solvers::thermal::Radiation parseBoundaryValue<solvers::thermal::Radiation>(const XMLReader& tag_with_value)
{
    return solvers::thermal::Radiation(tag_with_value.requireAttribute<double>("emissivity"), tag_with_value.requireAttribute<double>("ambient"));
}

} // namespace plask

#endif

