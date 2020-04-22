/**
 * \file ferminew.h
 * Sample solver header for your solver
 */
#ifndef PLASK__SOLVER_GAIN_FermiNew_H
#define PLASK__SOLVER_GAIN_FermiNew_H

#include <plask/plask.hpp>
#include "kublybr.h"

namespace plask { namespace solvers { namespace FermiNew {

template <typename GeometryT, typename T> struct DataBase;
template <typename GeometryT> struct GainData;
template <typename GeometryT> struct DgDnData;
template <typename GeometryT> struct LuminescenceData;

template <typename GeometryT> struct GainSpectrum;
template <typename GeometryT> struct LuminescenceSpectrum;

struct Levels {
    int mEc, mEvhh, mEvlh;          // to choose the correct band edges
    std::vector<warstwa*> mpEc, mpEvhh, mpEvlh;
    std::unique_ptr<struktura> mpStrEc, mpStrEvhh, mpStrEvlh;
    plask::shared_ptr<obszar_aktywny> aktyw;
    bool invalid;
    Levels(): mEc(-1), mEvhh(-1), mEvlh(-1), invalid(true) {}
    Levels(Levels&& src):
        mEc(src.mEc), mEvhh(src.mEvhh), mEvlh(src.mEvlh),
        mpEc(std::move(src.mpEc)), mpEvhh(std::move(src.mpEvhh)), mpEvlh(std::move(src.mpEvlh)),
        mpStrEc(std::move(src.mpStrEc)), mpStrEvhh(std::move(src.mpStrEvhh)), mpStrEvlh(std::move(src.mpStrEvlh)),
        aktyw(std::move(src.aktyw)), invalid(src.invalid) {
        src.mpEc.clear(); src.mpEvhh.clear(); src.mpEvlh.clear();
    }
    void clearEc() { for (warstwa* ptr: mpEc) delete ptr; mpEc.clear(); }
    void clearEvhh() { for (warstwa* ptr: mpEvhh) delete ptr; mpEvhh.clear(); }
    void clearEvlh() { for (warstwa* ptr: mpEvlh) delete ptr; mpEvlh.clear(); }
    ~Levels() { clearEc(); clearEvhh(); clearEvlh(); }
};

inline static double nm_to_eV(double wavelength) {
    return (plask::phys::h_eV * plask::phys::c) / (wavelength*1e-9);
}

/**
 * Gain solver using Fermi Golden Rule
 */
template <typename GeometryType>
struct PLASK_SOLVER_API FermiNewGainSolver: public SolverWithMesh<GeometryType,MeshAxis>
{
    /// Structure containing information about each active region
    struct ActiveRegionInfo
    {
        shared_ptr<StackContainer<2>> layers;   ///< Stack containing all layers in the active region
        Vec<2> origin;                          ///< Location of the active region stack origin
        ActiveRegionInfo(Vec<2> origin): layers(plask::make_shared<StackContainer<2>>()), origin(origin) {}

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
                        throw Exception("{0}: Multiple quantum well materials in active region.", solver->getId());*/
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
                        throw Exception("{0}: Multiple barrier materials around active region.", solver->getId());*/
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

    friend struct DataBase<GeometryType, Tensor2<double>>;
    friend struct DataBase<GeometryType, double>;
    friend struct GainData<GeometryType>;
    friend struct DgDnData<GeometryType>;
    friend struct LuminescenceData<GeometryType>;

    std::vector<Levels> region_levels;
    
    friend struct GainSpectrum<GeometryType>;
    friend struct LuminescenceSpectrum<GeometryType>;
    friend class wzmocnienie;

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

    void showEnergyLevels(std::string str, const std::unique_ptr<struktura>& structure, double nQW);

    wzmocnienie getGainModule(double wavelength, double T, double n, const ActiveRegionInfo& region, 
                           const Levels& levels, bool iShowSpecLogs=false);

    void prepareLevels(wzmocnienie& gmodule, const ActiveRegionInfo& region) {
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
    const LazyData<Tensor2<double>> getGain(Gain::EnumType what, const shared_ptr<const MeshD<2> > &dst_mesh, double wavelength, InterpolationMethod interp=INTERPOLATION_DEFAULT);

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
    Levels levels;                      ///< Computed energy levels
    std::unique_ptr<wzmocnienie> gMod;

    GainSpectrum(FermiNewGainSolver<GeometryT>* solver, const Vec<2> point):
        solver(solver), point(point)
    {
        auto mesh = plask::make_shared<const OnePointMesh<2>>(point);
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
        throw BadInput(solver->getId(), "Point {0} does not belong to any active region", point);
    }

    GainSpectrum(const GainSpectrum& orig):
        solver(orig.solver), point(orig.point), region(orig.region), T(orig.T), n(orig.n) {}

    GainSpectrum(GainSpectrum&& orig) = default;
    
    void onTChange(ReceiverBase&, ReceiverBase::ChangeReason) {
        T = solver->inTemperature(plask::make_shared<const OnePointMesh<2>>(point))[0];
    }

    void onNChange(ReceiverBase&, ReceiverBase::ChangeReason) {
        n = solver->inCarriersConcentration(plask::make_shared<const OnePointMesh<2>>(point))[0];
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
        if (!gMod) {
            solver->findEnergyLevels(levels, *region, T, true);
            gMod.reset(new wzmocnienie(std::move(solver->getGainModule(wavelength, T, n, *region, levels, true))));
        }

        double E = nm_to_eV(wavelength);
        double QWfrac = region->qwtotallen / region->totallen;
        double tau = solver->getLifeTime();
        if (!tau) return ( gMod->wzmocnienie_calk_bez_splotu(E) / QWfrac ); //20.10.2014 adding lifetime
        else return ( gMod->wzmocnienie_calk_ze_splotem(E, phys::hb_eV*1e12/tau) / QWfrac ); //20.10.2014 adding lifetime
    }
};

inline static double sumLuminescence(wzmocnienie& gain, double wavelength) {
    double E = nm_to_eV(wavelength);
    double result = 0.;
    for(int nr_c = 0; nr_c <= (int) gain.pasma->pasmo_przew.size() - 1; nr_c++)
        for(int nr_v = 0; nr_v <= (int) gain.pasma->pasmo_wal.size() - 1; nr_v++)
            result += gain.spont_od_pary_pasm(E, nr_c, nr_v, 0);  // TODO: consider other polarization (now only TE)
    return result;
}

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
    Levels levels;                      ///< Computed energy levels
    std::unique_ptr<wzmocnienie> gMod;

    LuminescenceSpectrum(FermiNewGainSolver<GeometryT>* solver, const Vec<2> point):
        solver(solver), point(point)
    {
        auto mesh = plask::make_shared<const OnePointMesh<2>>(point);
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
        throw BadInput(solver->getId(), "Point {0} does not belong to any active region", point);
    }

    LuminescenceSpectrum(const LuminescenceSpectrum& orig):
        solver(orig.solver), point(orig.point), region(orig.region), T(orig.T), n(orig.n) {}

    LuminescenceSpectrum(LuminescenceSpectrum&& orig) = default;
        
    void onTChange(ReceiverBase&, ReceiverBase::ChangeReason) {
        T = solver->inTemperature(plask::make_shared<const OnePointMesh<2>>(point))[0];
    }

    void onNChange(ReceiverBase&, ReceiverBase::ChangeReason) {
        n = solver->inCarriersConcentration(plask::make_shared<const OnePointMesh<2>>(point))[0];
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
        if (!gMod) {
            solver->findEnergyLevels(levels, *region, T, true);
            gMod.reset(new wzmocnienie(std::move(solver->getGainModule(wavelength, T, n, *region, levels, true))));
        }
        double QWfrac = region->qwtotallen / region->totallen;
        return sumLuminescence(*gMod, wavelength) / QWfrac;
    }
};

}}} // namespace

#endif

