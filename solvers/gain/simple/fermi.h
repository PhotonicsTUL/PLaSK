/**
 * \file
 * Sample solver header for your solver
 */
#ifndef PLASK__SOLVER_GAIN_FERMI_H
#define PLASK__SOLVER_GAIN_FERMI_H

#include <plask/plask.hpp>
#include "gainQW.h"

namespace plask { namespace solvers { namespace fermi {

template <typename GeometryT> struct GainSpectrum;

/**
 * Gain solver using Fermi Golden Rule
 */
template <typename GeometryType>
struct FermiGainSolver: public SolverWithMesh<GeometryType,RectilinearMesh1D>
{
    /// Structure containing information about each active region
    struct ActiveRegionInfo
    {
        shared_ptr<StackContainer<2>> layers;   ///< Stack containing all layers in the active region
        Vec<2> origin;                          ///< Location of the active region stack origin
        ActiveRegionInfo(Vec<2> origin): layers(make_shared<StackContainer<2>>()), origin(origin) {}

        /// \return number of layers in the active region with surrounding barriers
        size_t size() const
        {
            return layers->getChildrenCount();
        }

        /// \return material of \p n-th layer
        shared_ptr<Material> getLayerMaterial(size_t n) const
        {
            auto block = static_cast<Block<2>*>(static_cast<Translation<2>*>(layers->getChildNo(n).get())->getChild().get());
            return block->getRepresentativeMaterial();
        }

        /// \return translated bounding box of \p n-th layer
        Box2D getLayerBox(size_t n) const
        {
            return static_cast<GeometryObjectD<2>*>(layers->getChildNo(n).get())->getBoundingBox() + origin;
        }

        /// \return \p true if given layer is quantum well
        bool isQW(size_t n) const
        {
            return static_cast<Translation<2>*>(layers->getChildNo(n).get())->getChild()->hasRole("QW");
        }

        /// \return bounding box of the whole active region
        Box2D getBoundingBox() const
        {
            return layers->getBoundingBox() + origin;
        }

        /// \return \p true if the point is in the active region
        bool contains(const Vec<2>& point) const {
            return getBoundingBox().contains(point);
        }
    };

    std::vector<ActiveRegionInfo> regions;  ///< List of active regions

    /// Receiver for temperature.
    ReceiverFor<Temperature,GeometryType> inTemperature;

    /// Receiver for carriers concentration in the active region
    ReceiverFor<CarriersConcentration,GeometryType> inCarriersConcentration;

    /// Provider for gain distribution
    typename ProviderFor<Gain,GeometryType>::Delegate outGain;

    /// Provider for gain over carriers concentration derivative distribution
    typename ProviderFor<GainOverCarriersConcentration, GeometryType>::Delegate outGainOverCarriersConcentration;

    FermiGainSolver(const std::string& name="");

    virtual ~FermiGainSolver();

    virtual std::string getClassName() const;

    virtual void loadConfiguration(plask::XMLReader& reader, plask::Manager& manager);

    /// Function computing energy levels
    //  TODO: it should return computed levels
    void determineLevels(double T, double n);

  protected:

    friend struct GainSpectrum<GeometryType>;
    friend class QW::gain;

    double cond_waveguide_depth;///< waveguide conduction band depth [eV]
    double vale_waveguide_depth;///< waveguide valence band depth [eV]
    double mLifeTime;///< stimulated emission lifetime [ps]
    double mMatrixElem;///< optical matrix element [m0*eV]
    double differenceQuotient;///< difference quotient of dG_dn derivative

    DataVector<const double> nOnMesh; // carriers concentration on the mesh
    DataVector<const double> TOnMesh;

//    double lambda_start;
//    double lambda_stop;
//    double lambda;

    QW::gain getGainModule(double wavelength, double T, double n, const ActiveRegionInfo& active);

    double nm_to_eV(double wavelength) {
        return (plask::phys::h_eV*plask::phys::c)/(wavelength*1e-9);
    }

    /// Initialize the solver
    virtual void onInitialize();

    /// Invalidate the gain
    virtual void onInvalidate();

    /// Notify that gain was chaged
    void onInputChange(ReceiverBase&, ReceiverBase::ChangeReason)
    {
        outGain.fireChanged();  // the input changed, so we inform the world that everybody should get the new gain
    }

    /**
     * Detect active regions.
     * Store information about them in the \p regions field.
     */
    void detectActiveRegions();

    /**
     * Compute width of the box
     * \param materialBox box to compute the width of
     * \return width of the box
     */
    double determineBoxWidth(plask::Box2D materialBox)
    {
        return  materialBox.upper[1] - materialBox.lower[1];
    }

    /**
     * Method computing the gain on the mesh (called by gain provider)
     * \param dst_mesh destination mesh
     * \param wavelength wavelength to compute gain for
     * \return gain distribution
     */
    const DataVector<double> getGain(const MeshD<2>& dst_mesh, double wavelength, InterpolationMethod=INTERPOLATION_DEFAULT);
    const DataVector<double> getdGdn(const MeshD<2>& dst_mesh, double wavelength, InterpolationMethod=INTERPOLATION_DEFAULT);

  public:
    double getLifeTime() const { return mLifeTime; }
    void setLifeTime(double iLifeTime)  { mLifeTime = iLifeTime; }

    double getMatrixElem() const { return mMatrixElem; }
    void setMatrixElem(double iMatrixElem)  { mMatrixElem = iMatrixElem; }

    /**
     * Reg gain spectrum object for future use;
     */
    GainSpectrum<GeometryType> getGainSpectrum(const Vec<2>& point);
};


/**
 * Cached gain spectrum
 */
template <typename GeometryT>
struct GainSpectrum {

    FermiGainSolver<GeometryT>* solver; ///< Source solver
    Vec<2> point;                       ///< Point in which the gain is calculated

    /// Active region containg the point
    const typename FermiGainSolver<GeometryT>::ActiveRegionInfo* region;

    double T;                           ///< Temperature
    double n;                           ///< Carries concentration

    GainSpectrum(FermiGainSolver<GeometryT>* solver, const Vec<2> point): solver(solver), point(point), T(NAN), n(NAN) {
        for (const auto& reg: solver->regions) {
            if (reg.contains(point)) {
                region = &reg;
                return;
            };
        }
        throw BadInput(solver->getId(), "Point %1% does not belong to any active region", point);
    }

    /**
     * Get gain at given valenegth
     * \param wavelength wavelength to get gain
     * \return gain
     */
    double getGain(double wavelength) {
        if (isnan(T) || solver->inTemperature.changed())
            T = solver->inTemperature(OnePointMesh<2>(point))[0];
        if (isnan(n) || solver->inCarriersConcentration.changed())
            n = solver->inCarriersConcentration(OnePointMesh<2>(point))[0];
        return solver->getGainModule(wavelength, T, n, *region).Get_gain_at(solver->nm_to_eV(wavelength));
    }
};




}}} // namespace

#endif

