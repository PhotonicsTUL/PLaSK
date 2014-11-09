/**
 * \file
 * Sample solver header for your solver
 */
#ifndef PLASK__SOLVER_GAIN_FERMINEW_H
#define PLASK__SOLVER_GAIN_FERMINEW_H

#include <plask/plask.hpp>
#include "kubly.h"

namespace plask { namespace solvers { namespace ferminew {

template <typename GeometryT> struct GainSpectrum;
template <typename GeometryT> struct LuminescenceSpectrum;

/**
 * Gain solver using Fermi Golden Rule
 */
template <typename GeometryType>
struct PLASK_SOLVER_API FerminewGainSolver: public SolverWithMesh<GeometryType,OrderedMesh1D>
{
    /// Structure containing information about each active region
    struct ActiveRegionInfo
    {
        shared_ptr<StackContainer<2>> layers;   ///< Stack containing all layers in the active region
        Vec<2> origin;                          ///< Location of the active region stack origin
        ActiveRegionInfo(Vec<2> origin): layers(make_shared<StackContainer<2>>()), origin(origin) {}

        /// Return number of layers in the active region with surrounding barriers
        size_t size() const
        {
            return layers->getChildrenCount();
        }

        /// Return material of \p n-th layer
        shared_ptr<Material> getLayerMaterial(size_t n) const
        {
            auto block = static_cast<Block<2>*>(static_cast<Translation<2>*>(layers->getChildNo(n).get())->getChild().get());
            if (auto m = block->singleMaterial()) return m;
            throw plask::Exception("FerminewGainSolver requires solid layers.");
        }

        /// Return translated bounding box of \p n-th layer
        Box2D getLayerBox(size_t n) const
        {
            return static_cast<GeometryObjectD<2>*>(layers->getChildNo(n).get())->getBoundingBox() + origin;
        }

        /// Return \p true if given layer is quantum well
        bool isQW(size_t n) const
        {
            return static_cast<Translation<2>*>(layers->getChildNo(n).get())->getChild()->hasRole("QW");
        }

        /// Return bounding box of the whole active region
        Box2D getBoundingBox() const
        {
            return layers->getBoundingBox() + origin;
        }

        /// Return \p true if the point is in the active region
        bool contains(const Vec<2>& point) const {
            return getBoundingBox().contains(point);
        }

        // LUKASZ
        std::vector< shared_ptr<Material> > actMaterials; ///< All materials in the active region
        std::vector<double> lens; ///< Thicknesses of the layers in the active region

        shared_ptr<Material> materialQW;        ///< Quantum well material
        shared_ptr<Material> materialBarrier;   ///< Barrier material
        double qwlen;                           ///< Single quantum well thickness [Å]
        double qwtotallen;                      ///< Total quantum wells thickness [Å]
        double totallen;                        ///< Total active region thickness [Å]
        double bottomlen;                       ///< Bottom spacer thickness [Å]
        double toplen;                          ///< Top spacer thickness [Å]

        /**
         * Summarize active region, check for appropriateness and compute some values
         * \param solver solver
         */
        void summarize(const FerminewGainSolver<GeometryType>* solver) {
            auto bbox = layers->getBoundingBox();
            totallen = 1e4 * (bbox.upper[1] - bbox.lower[1] - bottomlen - toplen);  // 1e4: µm -> Å
            size_t qwn = 0;
            qwtotallen = 0.;
            bool lastbarrier = true;
            for (const auto& layer: layers->children) {
                auto block = static_cast<Block<2>*>(static_cast<Translation<2>*>(layer.get())->getChild().get());
                auto material = block->singleMaterial();
                if (!material) throw plask::Exception("FerminewGainSolver requires solid layers.");
                if (static_cast<Translation<2>*>(layer.get())->getChild()->hasRole("QW"))
                {
                    if (!materialQW)
                        materialQW = material;
                    else if (*material != *materialQW)
                        throw Exception("%1%: Multiple quantum well materials in active region.", solver->getId());
                    auto bbox = static_cast<GeometryObjectD<2>*>(layer.get())->getBoundingBox();
                    qwtotallen += bbox.upper[1] - bbox.lower[1];
                    if (lastbarrier) ++qwn;
                    else solver->writelog(LOG_WARNING, "Considering two adjacent quantum wells as one");
                    lastbarrier = false;
                }
                else //if (static_cast<Translation<2>*>(layer.get())->getChild()->hasRole("barrier")) // TODO 4.09.2014
                {
                    if (!materialBarrier)
                        materialBarrier = material;
                    /*else if (!is_zero(material->Me(300).c00 - materialBarrier->Me(300).c00) ||
                             !is_zero(material->Me(300).c11 - materialBarrier->Me(300).c11) ||
                             !is_zero(material->Mhh(300).c00 - materialBarrier->Mhh(300).c00) ||
                             !is_zero(material->Mhh(300).c11 - materialBarrier->Mhh(300).c11) ||
                             !is_zero(material->Mlh(300).c00 - materialBarrier->Mlh(300).c00) ||
                             !is_zero(material->Mlh(300).c11 - materialBarrier->Mlh(300).c11) ||
                             !is_zero(material->CB(300) - materialBarrier->CB(300)) ||
                             !is_zero(material->VB(300) - materialBarrier->VB(300)))
                        throw Exception("%1%: Multiple barrier materials around active region.", solver->getId());*/
                    lastbarrier = true;
                } // TODO something must be added here because of spacers placed next to external barriers
            }
            qwtotallen *= 1e4; // µm -> Å
            qwlen = qwtotallen / qwn;
        }
    };

    shared_ptr<Material> materialSubstrate;   ///< Substrate material

    ///< List of active regions
    std::vector<ActiveRegionInfo> regions;

    /// Receiver for temperature.
    ReceiverFor<Temperature,GeometryType> inTemperature;

    /// Receiver for carriers concentration in the active region
    ReceiverFor<CarriersConcentration,GeometryType> inCarriersConcentration;

    /// Provider for gain distribution
    typename ProviderFor<Gain,GeometryType>::Delegate outGain;

    /// Provider for luminescence distribution
    typename ProviderFor<Luminescence,GeometryType>::Delegate outLuminescence;

    /// Provider for gain over carriers concentration derivative distribution
    typename ProviderFor<GainOverCarriersConcentration, GeometryType>::Delegate outGainOverCarriersConcentration;

    FerminewGainSolver(const std::string& name="");

    virtual ~FerminewGainSolver();

    virtual std::string getClassName() const;

    virtual void loadConfiguration(plask::XMLReader& reader, plask::Manager& manager);

  protected:

    friend struct GainSpectrum<GeometryType>;
    friend struct LuminescenceSpectrum<GeometryType>;
    friend class QW::gain;

    double cond_qw_shift;           ///< additional conduction band shift for qw [eV]
    double vale_qw_shift;           ///< additional valence band shift for qw [eV]
    double qw_width_mod;            ///< qw width modifier [-]
    double roughness;               ///< roughness [-]
    double lifetime;                ///< lifetime [ps]
    double matrixelem;              ///< optical matrix element [m0*eV]
    double differenceQuotient;      ///< difference quotient of dG_dn derivative
    bool fixQWsWidths;              ///< if true QW widths will not be changed for gain calculations

    int mEc, mEvhh, mEvlh; // to choose the correct band edges
    std::vector<QW::warstwa *> mpEc, mpEvhh, mpEvlh;
    QW::warstwa *mpLay;
    QW::struktura *mpStrEc, *mpStrEvhh, *mpStrEvlh;
    int buildStructure(double T, const ActiveRegionInfo& region, bool iShowSpecLogs=false);
    int buildEc(double T, const ActiveRegionInfo& region, bool iShowSpecLogs=false);
    int buildEvhh(double T, const ActiveRegionInfo& region, bool iShowSpecLogs=false);
    int buildEvlh(double T, const ActiveRegionInfo& region, bool iShowSpecLogs=false);
    //double cutNumber(double iNumber, int iN);
    double recalcConc(plask::shared_ptr<QW::obszar_aktywny> iAktyw, double iN, double iQWTotH, double iT, double iQWnR);

    DataVector<const double> nOnMesh; // carriers concentration on the mesh
    DataVector<const double> TOnMesh;

//    double lambda_start;
//    double lambda_stop;
//    double lambda;

    QW::gain getGainModule(double wavelength, double T, double n, const ActiveRegionInfo& region, bool iShowSpecLogs=false);

    void prepareLevels(QW::gain& gmodule, const ActiveRegionInfo& region) {
    }

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
        outLuminescence.fireChanged();  // the input changed, so we inform the world that everybody should get the new luminescence
    }

    /**
     * Detect active regions.
     * Store information about them in the \p regions field.
     */
    void detectActiveRegions();

    /**
     * Method computing the gain on the mesh (called by gain provider)
     * \param dst_mesh destination mesh
     * \param wavelength wavelength to compute gain for
     * \param interp interpolation method
     * \return gain distribution
     */
    const LazyData<double> getGain(const shared_ptr<const MeshD<2> > &dst_mesh, double wavelength, InterpolationMethod interp=INTERPOLATION_DEFAULT);
    const LazyData<double> getLuminescence(const shared_ptr<const MeshD<2> > &dst_mesh, double wavelength, InterpolationMethod interp=INTERPOLATION_DEFAULT);
    //const DataVector<double> getdGdn(const MeshD<2>& dst_mesh, double wavelength, InterpolationMethod interp=INTERPOLATION_DEFAULT); // LUKASZ

  public:

    bool if_strain;                 ///< Consider strain in QW?
    bool if_fixed_QWs_widths;    ///< Fix QWs widhts?

    double getRoughness() const { return roughness; }
    void setRoughness(double iRoughness)  { roughness = iRoughness; }

    double getLifeTime() const { return lifetime; }
    void setLifeTime(double iLifeTime)  { lifetime = iLifeTime; }

    double getMatrixElem() const { return matrixelem; }
    void setMatrixElem(double iMatrixElem)  { matrixelem = iMatrixElem; }

    double getCondQWShift() const { return cond_qw_shift; }
    void setCondQWShift(double iCondQWShift)  { cond_qw_shift = iCondQWShift; }

    double getValeQWShift() const { return vale_qw_shift; }
    void setValeQWShift(double iValeQWShift)  { vale_qw_shift = iValeQWShift; }

    /**
     * Reg gain spectrum object for future use;
     */
    GainSpectrum<GeometryType> getGainSpectrum(const Vec<2>& point);

    /**
     * Reg luminescence spectrum object for future use;
     */
    LuminescenceSpectrum<GeometryType> getLuminescenceSpectrum(const Vec<2>& point);
};


/**
 * Cached gain spectrum
 */
template <typename GeometryT>
struct GainSpectrum {

    FerminewGainSolver<GeometryT>* solver; ///< Source solver
    Vec<2> point;                       ///< Point in which the gain is calculated

    /// Active region containing the point
    const typename FerminewGainSolver<GeometryT>::ActiveRegionInfo* region;

    double T;                           ///< Temperature
    double n;                           ///< Carriers concentration
    QW::gain gMod; // added
    bool gModExist; // added

    GainSpectrum(FerminewGainSolver<GeometryT>* solver, const Vec<2> point): solver(solver), point(point), T(NAN), n(NAN)
    {
        //std::cout << "Setting gModExist to false\n"; // added
        gModExist = false; // added
    //GainSpectrum(FerminewGainSolver<GeometryT>* solver, const Vec<2> point): solver(solver), point(point), T(NAN), n(NAN) {
        for (const auto& reg: solver->regions) {
            if (reg.contains(point)) {
                region = &reg;
                solver->inTemperature.changedConnectMethod(this, &GainSpectrum::onTChange);
                solver->inCarriersConcentration.changedConnectMethod(this, &GainSpectrum::onNChange);
                return;
            };
        }
        throw BadInput(solver->getId(), "Point %1% does not belong to any active region", point);
    }

    void onTChange(ReceiverBase&, ReceiverBase::ChangeReason) { T = NAN; }

    void onNChange(ReceiverBase&, ReceiverBase::ChangeReason) { n = NAN; }

    ~GainSpectrum() {
        solver->inTemperature.changedDisconnectMethod(this, &GainSpectrum::onTChange);
        solver->inCarriersConcentration.changedDisconnectMethod(this, &GainSpectrum::onNChange);
    }

    /**
     * Get gain at given valenegth
     * \param wavelength wavelength to get gain
     * \return gain
     */
    double getGain(double wavelength)
    {
        //std::cout << "Runing getGain in spectrum\n"; // added
        //double getGain(double wavelength) {
        if (isnan(T)) T = solver->inTemperature(make_shared<const OnePointMesh<2>>(point))[0];
        if (isnan(n)) n = solver->inCarriersConcentration(make_shared<const OnePointMesh<2>>(point))[0];
        if (!gModExist) // added
        { // added
            //std::cout << "Getting GainModule in spectrum\n"; // added
            gMod = solver->getGainModule(wavelength, T, n, *region, true); // added
            gModExist = true; // added
        } // added
        return gMod.Get_gain_at_n(solver->nm_to_eV(wavelength), region->qwtotallen, region->qwtotallen / region->totallen, solver->getLifeTime()); // added
        //return solver->getGainModule(wavelength, T, n, *region) // commented
        //    .Get_gain_at_n(solver->nm_to_eV(wavelength), region->qwtotallen, region->qwtotallen / region->totallen); // commented
        /*return solver->getGainModule(wavelength, T, n, *region)
            .Get_gain_at_n(solver->nm_to_eV(wavelength), region->qwtotallen, region->qwtotallen / region->totallen);*/
    }
};

/**
 * Cached luminescence spectrum
 */
template <typename GeometryT>
struct LuminescenceSpectrum {

    FerminewGainSolver<GeometryT>* solver; ///< Source solver
    Vec<2> point;                       ///< Point in which the luminescence is calculated

    /// Active region containing the point
    const typename FerminewGainSolver<GeometryT>::ActiveRegionInfo* region;

    double T;                           ///< Temperature
    double n;                           ///< Carriers concentration
    QW::gain gMod; // added
    bool gModExist; // added

    LuminescenceSpectrum(FerminewGainSolver<GeometryT>* solver, const Vec<2> point): solver(solver), point(point), T(NAN), n(NAN)
    {
        //std::cout << "Setting gModExist to false\n"; // added
        gModExist = false; // added
    //LuminescenceSpectrum(FerminewGainSolver<GeometryT>* solver, const Vec<2> point): solver(solver), point(point), T(NAN), n(NAN) {
        for (const auto& reg: solver->regions) {
            if (reg.contains(point)) {
                region = &reg;
                solver->inTemperature.changedConnectMethod(this, &LuminescenceSpectrum::onTChange);
                solver->inCarriersConcentration.changedConnectMethod(this, &LuminescenceSpectrum::onNChange);
                return;
            };
        }
        throw BadInput(solver->getId(), "Point %1% does not belong to any active region", point);
    }

    void onTChange(ReceiverBase&, ReceiverBase::ChangeReason) { T = NAN; }

    void onNChange(ReceiverBase&, ReceiverBase::ChangeReason) { n = NAN; }

    ~LuminescenceSpectrum() {
        solver->inTemperature.changedDisconnectMethod(this, &LuminescenceSpectrum::onTChange);
        solver->inCarriersConcentration.changedDisconnectMethod(this, &LuminescenceSpectrum::onNChange);
    }

    /**
     * Get luminescence at given valenegth
     * \param wavelength wavelength to get luminescence
     * \return luminescence
     */
    double getLuminescence(double wavelength)
    {
        //std::cout << "Runing getLuminescence in spectrum\n"; // added
    //double getLuminescence(double wavelength) {
        if (isnan(T)) T = solver->inTemperature(make_shared<const OnePointMesh<2>>(point))[0];
        if (isnan(n)) n = solver->inCarriersConcentration(make_shared<const OnePointMesh<2>>(point))[0];
        if (!gModExist) // added
        { // added
            //std::cout << "Getting GainModule in spectrum\n"; // added
            gMod = solver->getGainModule(wavelength, T, n, *region, true); // added
            gModExist = true; // added
        } // added
        return gMod.Get_luminescence_at_n(solver->nm_to_eV(wavelength), region->qwtotallen, region->qwtotallen / region->totallen); // added
        //return solver->getGainModule(wavelength, T, n, *region) // commented
        //    .Get_luminescence_at_n(solver->nm_to_eV(wavelength), region->qwtotallen, region->qwtotallen / region->totallen); // commented
        /*return solver->getGainModule(wavelength, T, n, *region)
            .Get_luminescence_at_n(solver->nm_to_eV(wavelength), region->qwtotallen, region->qwtotallen / region->totallen);*/
    }
};

}}} // namespace

#endif

