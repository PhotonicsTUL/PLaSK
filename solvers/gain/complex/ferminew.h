/**
 * \file
 * Sample solver header for your solver
 */
#ifndef PLASK__SOLVER_GAIN_FermiNew_H
#define PLASK__SOLVER_GAIN_FermiNew_H

#include <plask/plask.hpp>
#include "kubly.h"

namespace plask { namespace solvers { namespace FermiNew {

template <typename GeometryT> struct GainSpectrum;
template <typename GeometryT> struct LuminescenceSpectrum;

/**
 * Gain solver using Fermi Golden Rule
 */
template <typename GeometryType>
struct PLASK_SOLVER_API FermiNewGainSolver: public SolverWithMesh<GeometryType,OrderedMesh1D>
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
            throw plask::Exception("FermiNewGainSolver requires solid layers.");
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

        /// Return \p true if given point is inside quantum well
        bool inQW(const Vec<2>& point) const {
            if (!contains(point)) return false;
            assert(layers->getChildForHeight(point.c1-origin.c1));
            return layers->getChildForHeight(point.c1-origin.c1)->getChild()->hasRole("QW");
        }

        // LUKASZ
        std::vector< shared_ptr<Material> > actMaterials; ///< All materials in the active region
        std::vector<double> lens; ///< Thicknesses of the layers in the active region

        //shared_ptr<Material> materialQW;        ///< Quantum well material
        //shared_ptr<Material> materialBarrier;   ///< Barrier material
        double qwlen;                           ///< Single quantum well thickness [Å]
        double qwtotallen;                      ///< Total quantum wells thickness [Å]
        double totallen;                        ///< Total active region thickness [Å]
        double bottomlen;                       ///< Bottom spacer thickness [Å]
        double toplen;                          ///< Top spacer thickness [Å]

        /**
         * Summarize active region, check for appropriateness and compute some values
         * \param solver solver
         */
        void summarize(const FermiNewGainSolver<GeometryType>* solver) {
            auto bbox = layers->getBoundingBox();
            totallen = 1e4 * (bbox.upper[1] - bbox.lower[1] - bottomlen - toplen);  // 1e4: µm -> Å
            size_t qwn = 0;
            qwtotallen = 0.;
            bool lastbarrier = true;
            for (const auto& layer: layers->children) {
                auto block = static_cast<Block<2>*>(static_cast<Translation<2>*>(layer.get())->getChild().get());
                auto material = block->singleMaterial();
                if (!material) throw plask::Exception("FermiNewGainSolver requires solid layers.");
                if (static_cast<Translation<2>*>(layer.get())->getChild()->hasRole("QW"))
                {
                    /*if (!materialQW)
                        materialQW = material;
                    else if (*material != *materialQW)
                        throw Exception("%1%: Multiple quantum well materials in active region.", solver->getId());*/
                    auto bbox = static_cast<GeometryObjectD<2>*>(layer.get())->getBoundingBox();
                    qwtotallen += bbox.upper[1] - bbox.lower[1];
                    if (lastbarrier) ++qwn;
                    else solver->writelog(LOG_WARNING, "Considering two adjacent quantum wells as one");
                    lastbarrier = false;
                }
                else //if (static_cast<Translation<2>*>(layer.get())->getChild()->hasRole("barrier")) // TODO 4.09.2014
                {
                    /*if (!materialBarrier)
                        materialBarrier = material;
                    else if (!is_zero(material->Me(300).c00 - materialBarrier->Me(300).c00) ||
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

    FermiNewGainSolver(const std::string& name="");

    virtual ~FermiNewGainSolver();

    virtual std::string getClassName() const;

    virtual void loadConfiguration(plask::XMLReader& reader, plask::Manager& manager);

    struct DataBase;
    struct GainData;
    struct DgDnData;
    struct LuminescenceData;

    struct Levels {
        int mEc, mEvhh, mEvlh;          // to choose the correct band edges
        std::vector<std::unique_ptr<QW::Warstwa>> mpEc, mpEvhh, mpEvlh;
        std::unique_ptr<QW::Struktura> mpStrEc, mpStrEvhh, mpStrEvlh;
        plask::shared_ptr<QW::ObszarAktywny> aktyw;
        bool invalid;
        Levels(): mEc(-1), mEvhh(-1), mEvlh(-1), invalid(true) {}
    };

    std::vector<Levels> region_levels;
    
    friend struct GainSpectrum<GeometryType>;
    friend struct LuminescenceSpectrum<GeometryType>;
    friend class QW::Gain;

    double cond_qw_shift;           ///< additional conduction band shift for qw [eV]
    double vale_qw_shift;           ///< additional valence band shift for qw [eV]
    double qw_width_mod;            ///< qw width modifier [-]
    double roughness;               ///< roughness [-]
    double lifetime;                ///< lifetime [ps]
    double matrix_elem;             ///< optical matrix element [m0*eV]
    double matrix_elem_sc_fact;     ///< scaling factor for optical matrix element [-]
    double differenceQuotient;      ///< difference quotient of dG_dn derivative
    //bool fixQWsWidths;            ///< if true QW widths will not be changed for gain calculations
    double Tref;                    ///< reference temperature [K]                                          // 11.12.2014 - dodana linia

    void findEnergyLevels(Levels& levels, const ActiveRegionInfo& region, double iT, bool iShowSpecLogs=false);
    int buildStructure(Levels& levels, double T, const ActiveRegionInfo& region, bool iShowSpecLogs=false);
    int buildEc(Levels& levels, double T, const ActiveRegionInfo& region, bool iShowSpecLogs=false);
    int buildEvhh(Levels& levels, double T, const ActiveRegionInfo& region, bool iShowSpecLogs=false);
    int buildEvlh(Levels& levels, double T, const ActiveRegionInfo& region, bool iShowSpecLogs=false);

    QW::Gain getGainModule(double wavelength, double T, double n, const ActiveRegionInfo& region, 
                           const Levels& levels, bool iShowSpecLogs=false);

    void prepareLevels(QW::Gain& gmodule, const ActiveRegionInfo& region) {
    }

    inline static double nm_to_eV(double wavelength) {
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
    const LazyData<double> getGain(Gain::EnumType what, const shared_ptr<const MeshD<2> > &dst_mesh, double wavelength, InterpolationMethod interp=INTERPOLATION_DEFAULT);

    const LazyData<double> getLuminescence(const shared_ptr<const MeshD<2> > &dst_mesh, double wavelength, InterpolationMethod interp=INTERPOLATION_DEFAULT);


    bool strains;            ///< Consider strain in QWs and barriers?
    bool adjust_widths;      ///< Adjust widths of the QWs?
    bool build_struct_once;  ///< Build active-region structure only once?

  public:

    bool getStrains() const { return strains; }
    void setStrains(bool value)  {
        if (strains != value) {
            strains = value;
            if (build_struct_once) this->invalidate();
        }
    }

    bool getAdjustWidths() const { return adjust_widths; }
    void setAdjustWidths(bool value)  {
        if (adjust_widths != value) {
            adjust_widths = value;
            this->invalidate();
        }
    }

    bool getBuildStructOnce() const { return build_struct_once; }
    void setBuildStructOnce(bool value)  {
        if (build_struct_once != value) {
            build_struct_once = value;
            this->invalidate();
        }
    }

    double getRoughness() const { return roughness; }
    void setRoughness(double iRoughness)  {
        if (roughness != iRoughness) {
            roughness = iRoughness;
            if (build_struct_once) this->invalidate();
        }
    }

    double getLifeTime() const { return lifetime; }
    void setLifeTime(double iLifeTime)  {
        if (lifetime != iLifeTime) {
            lifetime = iLifeTime;
            if (build_struct_once) this->invalidate();
        }
    }

    double getMatrixElem() const { return matrix_elem; }
    void setMatrixElem(double iMatrixElem)  {
        if (matrix_elem != iMatrixElem) {
            matrix_elem = iMatrixElem;
            if (build_struct_once) this->invalidate();
        }
    }

    double getMatrixElemScFact() const { return matrix_elem_sc_fact; }
    void setMatrixElemScFact(double iMatrixElemScFact)  {
        if (matrix_elem_sc_fact != iMatrixElemScFact) {
            matrix_elem_sc_fact = iMatrixElemScFact;
            if (build_struct_once) this->invalidate();
        }
    }

    double getCondQWShift() const { return cond_qw_shift; }
    void setCondQWShift(double iCondQWShift)  {
        if (cond_qw_shift != iCondQWShift) {
            cond_qw_shift = iCondQWShift;
            if (build_struct_once) this->invalidate();
        }
    }

    double getValeQWShift() const { return vale_qw_shift; }
    void setValeQWShift(double iValeQWShift)  {
        if (vale_qw_shift != iValeQWShift) {
            vale_qw_shift = iValeQWShift;
            if (build_struct_once) this->invalidate();
        }
    }

    double getTref() const { return Tref; }
    void setTref(double iTref)  { 
        if (Tref != iTref) {
            Tref = iTref;
            if (build_struct_once) this->invalidate();
        }
    }

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

    FermiNewGainSolver<GeometryT>* solver; ///< Source solver
    Vec<2> point;                       ///< Point in which the gain is calculated

    /// Active region containing the point
    const typename FermiNewGainSolver<GeometryT>::ActiveRegionInfo* region;

    double T;                           ///< Temperature
    double n;                           ///< Carriers concentration
    typename FermiNewGainSolver<GeometryT>::Levels levels; ///< Computed energy levels
    QW::Gain gMod;
    bool gModExist;

    GainSpectrum(FermiNewGainSolver<GeometryT>* solver, const Vec<2> point):
        solver(solver), point(point), gModExist(false)
    {
        auto mesh = make_shared<const OnePointMesh<2>>(point);
        T = solver->inTemperature(mesh)[0];
        n = solver->inCarriersConcentration(mesh)[0];
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

    GainSpectrum(const GainSpectrum& orig):
        solver(orig.solver), point(orig.point), region(orig.region), T(orig.T), n(orig.n), gModExist(false) {}

    GainSpectrum(GainSpectrum&& orig) = default;
    
    void onTChange(ReceiverBase&, ReceiverBase::ChangeReason) {
        T = solver->inTemperature(make_shared<const OnePointMesh<2>>(point))[0];
        gModExist = false;
    }

    void onNChange(ReceiverBase&, ReceiverBase::ChangeReason) {
        n = solver->inCarriersConcentration(make_shared<const OnePointMesh<2>>(point))[0];
        gModExist = false;
    }

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
        if (!gModExist) {
            solver->findEnergyLevels(levels, *region, T, true);
            gMod = solver->getGainModule(wavelength, T, n, *region, levels, true);
            gModExist = true;
        }
        return gMod.Get_gain_at_n(solver->nm_to_eV(wavelength), region->qwtotallen, region->qwtotallen / region->totallen, solver->getLifeTime());
    }
};

/**
 * Cached luminescence spectrum
 */
template <typename GeometryT>
struct LuminescenceSpectrum {

    FermiNewGainSolver<GeometryT>* solver; ///< Source solver
    Vec<2> point;                       ///< Point in which the luminescence is calculated

    /// Active region containing the point
    const typename FermiNewGainSolver<GeometryT>::ActiveRegionInfo* region;

    double T;                           ///< Temperature
    double n;                           ///< Carriers concentration
    typename FermiNewGainSolver<GeometryT>::Levels levels; ///< Computed energy levels
    QW::Gain gMod;
    bool gModExist;

    LuminescenceSpectrum(FermiNewGainSolver<GeometryT>* solver, const Vec<2> point):
        solver(solver), point(point), gModExist(false)
    {
        auto mesh = make_shared<const OnePointMesh<2>>(point);
        T = solver->inTemperature(mesh)[0];
        n = solver->inCarriersConcentration(mesh)[0];
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

    LuminescenceSpectrum(const LuminescenceSpectrum& orig):
        solver(orig.solver), point(orig.point), region(orig.region), T(orig.T), n(orig.n), gModExist(false) {}

    LuminescenceSpectrum(LuminescenceSpectrum&& orig) = default;
        
    void onTChange(ReceiverBase&, ReceiverBase::ChangeReason) {
        T = solver->inTemperature(make_shared<const OnePointMesh<2>>(point))[0];
        gModExist = false;
    }

    void onNChange(ReceiverBase&, ReceiverBase::ChangeReason) {
        n = solver->inCarriersConcentration(make_shared<const OnePointMesh<2>>(point))[0];
        gModExist = false;
    }


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
        if (!gModExist) {
            solver->findEnergyLevels(levels, *region, T, true);
            gMod = solver->getGainModule(wavelength, T, n, *region, levels, true);
            gModExist = true;
        }
        return gMod.Get_luminescence_at_n(solver->nm_to_eV(wavelength), region->qwtotallen, region->qwtotallen / region->totallen); // added
    }
};

}}} // namespace

#endif

