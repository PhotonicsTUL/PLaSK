/*
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
/**
 * \file ferminew.h
 * Sample solver header for your solver
 */
#ifndef PLASK__SOLVER_GAIN_FermiNew_H
#define PLASK__SOLVER_GAIN_FermiNew_H

#include <plask/plask.hpp>
#include "wzmocnienie/kublybr.h"

namespace plask { namespace solvers { namespace FermiNew {

template <typename GeometryT, typename T> struct DataBase;
template <typename GeometryT> struct GainData;
template <typename GeometryT> struct DgDnData;
template <typename GeometryT> struct LuminescenceData;

template <typename GeometryT> struct GainSpectrum;
template <typename GeometryT> struct LuminescenceSpectrum;

struct Levels {
    double Eg;
    std::unique_ptr<kubly::struktura> bandsEc, bandsEvhh, bandsEvlh;
    std::unique_ptr<kubly::struktura> modbandsEc, modbandsEvhh, modbandsEvlh;
    plask::shared_ptr<kubly::obszar_aktywny> activeRegion;
    operator bool() const {
        return bool(bandsEc) || bool(bandsEvhh) || bool(bandsEvlh);
    }
};

inline static double nm_to_eV(double wavelength) { return (plask::phys::h_eV * plask::phys::c) / (wavelength * 1e-9); }

/**
 * Gain solver using Fermi Golden Rule
 */
template <typename GeometryType>
struct PLASK_SOLVER_API FermiNewGainSolver : public SolverWithMesh<GeometryType, MeshAxis> {
    /**
     *  Structure containing information about each active region
     */
    struct ActiveRegionData {
        shared_ptr<StackContainer<2>> layers;  ///< Stack containing all layers in the active region
        Vec<2> origin;                         ///< Location of the active region stack origin
        std::set<int> QWs;

        // LUKASZ
        std::vector<double> lens;  ///< Thicknesses of the layers in the active region

        // shared_ptr<Material> materialQW;        ///< Quantum well material
        // shared_ptr<Material> materialBarrier;   ///< Barrier material
        double qwlen;       ///< Single quantum well thickness [Å]
        double qwtotallen;  ///< Total quantum wells thickness [Å]
        double totallen;    ///< Total active region thickness [Å]

        ActiveRegionData(Vec<2> origin) : layers(plask::make_shared<StackContainer<2>>()), origin(origin) {}

        /// Return number of layers in the active region with surrounding barriers
        size_t size() const { return layers->getChildrenCount(); }

        /// Return material of \p n-th layer
        shared_ptr<Material> getLayerMaterial(size_t n) const {
            auto block =
                static_cast<Block<2>*>(static_cast<Translation<2>*>(layers->getChildNo(n).get())->getChild().get());
            if (auto m = block->singleMaterial()) return m;
            throw plask::Exception("FermiNewGainSolver requires solid layers.");
        }

        /// Return translated bounding box of \p n-th layer
        Box2D getLayerBox(size_t n) const {
            return static_cast<GeometryObjectD<2>*>(layers->getChildNo(n).get())->getBoundingBox() + origin;
        }

        /// Return \p true if given layer is quantum well
        bool isQW(size_t n) const {
            return static_cast<Translation<2>*>(layers->getChildNo(n).get())->getChild()->hasRole("QW");
        }

        /// Return bounding box of the whole active region
        Box2D getBoundingBox() const { return layers->getBoundingBox() + origin; }

        /// Return \p true if the point is in the active region
        bool contains(const Vec<2>& point) const { return getBoundingBox().contains(point); }

        /// Return \p true if given point is inside quantum well
        bool inQW(const Vec<2>& point) const {
            if (!contains(point)) return false;
            assert(layers->getChildForHeight(point.c1 - origin.c1));
            return layers->getChildForHeight(point.c1 - origin.c1)->getChild()->hasRole("QW");
        }

        /**
         * Summarize active region, check for appropriateness and compute some values
         * \param solver solver
         */
        void summarize(const FermiNewGainSolver<GeometryType>* solver) {
            totallen = 1e4 * (layers->getBoundingBox().height() -
                              static_pointer_cast<GeometryObjectD<GeometryType::DIM>>(layers->getChildNo(0))
                                  ->getBoundingBox()
                                  .height() -
                              static_pointer_cast<GeometryObjectD<GeometryType::DIM>>(
                                  layers->getChildNo(layers->getChildrenCount() - 1))
                                  ->getBoundingBox()
                                  .height());  // 1e4: µm -> Å
            size_t qwn = 0;
            qwtotallen = 0.;
            bool lastbarrier = true;
            for (const auto& layer : layers->children) {
                auto block = static_cast<Block<2>*>(static_cast<Translation<2>*>(layer.get())->getChild().get());
                auto material = block->singleMaterial();
                if (!material) throw plask::Exception("FermiNewGainSolver requires solid layers.");
                if (static_cast<Translation<2>*>(layer.get())->getChild()->hasRole("QW")) {
                    /*if (!materialQW)
                        materialQW = material;
                    else if (*material != *materialQW)
                        throw Exception("{0}: Multiple quantum well materials in active region.", solver->getId());*/
                    auto bbox = static_cast<GeometryObjectD<2>*>(layer.get())->getBoundingBox();
                    qwtotallen += bbox.upper[1] - bbox.lower[1];
                    if (lastbarrier)
                        ++qwn;
                    else
                        solver->writelog(LOG_WARNING, "Considering two adjacent quantum wells as one");
                    lastbarrier = false;
                } else  // if (static_cast<Translation<2>*>(layer.get())->getChild()->hasRole("barrier")) //
                        // TODO 4.09.2014
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
                }  // TODO something must be added here because of spacers placed next to external barriers
            }
            qwtotallen *= 1e4;  // µm -> Å
            qwlen = qwtotallen / qwn;
        }
    };

    /**
     * Active region information with optional modified structure
     */
    struct ActiveRegionInfo : public ActiveRegionData {
        boost::optional<ActiveRegionData> mod;  ///< Modified structure data

        ActiveRegionInfo(Vec<2> origin) : ActiveRegionData(origin) {}

        ActiveRegionInfo(const ActiveRegionData& src) : ActiveRegionData(src) {}

        ActiveRegionInfo(ActiveRegionData&& src) : ActiveRegionData(std::move(src)) {}
    };

    shared_ptr<GeometryType> geometry_mod;  ///< Modified geometry for broadening calculation

    shared_ptr<Material> materialSubstrate;  ///< Substrate material

    ///< List of active regions
    std::vector<ActiveRegionInfo> regions;

    /// Receiver for temperature.
    ReceiverFor<Temperature, GeometryType> inTemperature;

    /// Receiver for carriers concentration in the active region
    ReceiverFor<CarriersConcentration, GeometryType> inCarriersConcentration;

    /// Provider for gain distribution
    typename ProviderFor<Gain, GeometryType>::Delegate outGain;

    /// Provider for luminescence distribution
    typename ProviderFor<Luminescence, GeometryType>::Delegate outLuminescence;

    FermiNewGainSolver(const std::string& name = "");

    virtual ~FermiNewGainSolver();

    virtual std::string getClassName() const;

    virtual void loadConfiguration(plask::XMLReader& reader, plask::Manager& manager);

    /// Get modified geometry
    shared_ptr<GeometryType> getModGeometry() const { return geometry_mod; }

    /**
     * Set new modified geometry for the solver
     * @param geometry new modified geometry space
     */
    void setModGeometry(const shared_ptr<GeometryType>& geometry) {
        if (geometry == this->geometry_mod) return;
        writelog(LOG_INFO, "Attaching modified geometry to solver");
        disconnectModGeometry();
        this->geometry_mod = geometry;
        if (this->geometry_mod)
            this->geometry_mod->changedConnectMethod(this, &FermiNewGainSolver<GeometryType>::onModGeometryChange);
        onModGeometryChange(Geometry::Event(geometry.get(), 0));
    }

    void disconnectModGeometry() {
        if (this->geometry_mod)
            this->geometry_mod->changedDisconnectMethod(this, &FermiNewGainSolver<GeometryType>::onModGeometryChange);
    }

    friend struct DataBase<GeometryType, Tensor2<double>>;
    friend struct DataBase<GeometryType, double>;
    friend struct GainData<GeometryType>;
    friend struct DgDnData<GeometryType>;
    friend struct LuminescenceData<GeometryType>;

    std::vector<Levels> region_levels;

    friend struct GainSpectrum<GeometryType>;
    friend struct LuminescenceSpectrum<GeometryType>;
    friend class wzmocnienie;

    double condQWshift;         ///< additional conduction band shift for qw (eV)
    double valeQWshift;         ///< additional valence band shift for qw (eV)
    double QWwidthMod;          ///< qw width modifier (-)
    double roughness;           ///< roughness (-)
    double lifetime;            ///< lifetime [ps]
    double matrixElem;          ///< optical matrix element [m0*eV]
    double differenceQuotient;  ///< difference quotient of dG_dn derivative
    double Tref;                ///< reference temperature (K)

    void findEnergyLevels(Levels& levels, const ActiveRegionInfo& region, double T, bool showDetails = false);

    void buildStructure(double T,
                        const ActiveRegionData& region,
                        std::unique_ptr<kubly::struktura>& bandsEc,
                        std::unique_ptr<kubly::struktura>& bandsEvhh,
                        std::unique_ptr<kubly::struktura>& bandsEvlh,
                        bool showDetails = false);
    kubly::struktura* buildEc(double T, const ActiveRegionData& region, bool showDetails = false);
    kubly::struktura* buildEvhh(double T, const ActiveRegionData& region, bool showDetails = false);
    kubly::struktura* buildEvlh(double T, const ActiveRegionData& region, bool showDetails = false);

    void showEnergyLevels(std::string str, const std::unique_ptr<kubly::struktura>& structure, double nQW);

    kubly::wzmocnienie getGainModule(double wavelength,
                                     double T,
                                     double n,
                                     const ActiveRegionInfo& region,
                                     const Levels& levels,
                                     bool iShowSpecLogs = false);

    void prepareLevels(kubly::wzmocnienie& gmodule, const ActiveRegionInfo& region) {}

    /// Initialize the solver
    virtual void onInitialize();

    /// Invalidate the gain
    virtual void onInvalidate();

    /// otify that modified geometry was chaged
    void onModGeometryChange(const Geometry::Event&) { this->invalidate(); }

    /// Notify that gain was chaged
    void onInputChange(ReceiverBase&, ReceiverBase::ChangeReason) {
        outGain.fireChanged();  // the input changed, so we inform the world that everybody should get the new gain
        outLuminescence
            .fireChanged();  // the input changed, so we inform the world that everybody should get the new luminescence
    }

    /**
     * Detect active regions.
     */
    std::list<ActiveRegionData> detectActiveRegions(const shared_ptr<GeometryType>& geometry);

    /**
     * Detect active regions and store their info in the \p regions field.
     * Store information about them in the \p regions field.
     */
    void prepareActiveRegionsInfo();

    /**
     * Method computing the gain on the mesh (called by gain provider)
     * \param dst_mesh destination mesh
     * \param wavelength wavelength to compute gain for
     * \param interp interpolation method
     * \return gain distribution
     */
    const LazyData<Tensor2<double>> getGain(Gain::EnumType what,
                                            const shared_ptr<const MeshD<2>>& dst_mesh,
                                            double wavelength,
                                            InterpolationMethod interp = INTERPOLATION_DEFAULT);

    const LazyData<double> getLuminescence(const shared_ptr<const MeshD<2>>& dst_mesh,
                                           double wavelength,
                                           InterpolationMethod interp = INTERPOLATION_DEFAULT);

    bool strains;            ///< Consider strain in QWs and barriers?
    bool adjust_widths;      ///< Adjust widths of the QWs?
    bool build_struct_once;  ///< Build active-region structure only once?

  public:
    bool getStrains() const { return strains; }
    void setStrains(bool value) {
        if (strains != value) {
            strains = value;
            if (build_struct_once) this->invalidate();
        }
    }

    bool getAdjustWidths() const { return adjust_widths; }
    void setAdjustWidths(bool value) {
        if (adjust_widths != value) {
            adjust_widths = value;
            this->invalidate();
        }
    }

    bool getBuildStructOnce() const { return build_struct_once; }
    void setBuildStructOnce(bool value) {
        if (build_struct_once != value) {
            build_struct_once = value;
            this->invalidate();
        }
    }

    double getRoughness() const { return roughness; }
    void setRoughness(double value) {
        if (roughness != value) {
            roughness = value;
            if (build_struct_once) this->invalidate();
        }
    }

    double getLifeTime() const { return lifetime; }
    void setLifeTime(double value) {
        if (lifetime != value) {
            lifetime = value;
            // if (build_struct_once) this->invalidate();
        }
    }

    double getMatrixElem() const { return matrixElem; }
    void setMatrixElem(double value) {
        if (matrixElem != value) {
            matrixElem = value;
            if (build_struct_once) this->invalidate();
        }
    }

    double getCondQWShift() const { return condQWshift; }
    void setCondQWShift(double value) {
        if (condQWshift != value) {
            condQWshift = value;
            if (build_struct_once) this->invalidate();
        }
    }

    double getValeQWShift() const { return valeQWshift; }
    void setValeQWShift(double value) {
        if (valeQWshift != value) {
            valeQWshift = value;
            if (build_struct_once) this->invalidate();
        }
    }

    double getTref() const { return Tref; }
    void setTref(double value) {
        if (Tref != value) {
            Tref = value;
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
template <typename GeometryT> struct PLASK_SOLVER_API GainSpectrum {
    FermiNewGainSolver<GeometryT>* solver;  ///< Source solver
    Vec<2> point;                           ///< Point in which the gain is calculated

    /// Active region containing the point
    size_t reg;

    double T;                   ///< Temperature
    double n;                   ///< Carriers concentration
    unique_ptr<Levels> levels;  ///< Computed energy levels
    std::unique_ptr<kubly::wzmocnienie> gMod;

    GainSpectrum(FermiNewGainSolver<GeometryT>* solver, const Vec<2> point);

    GainSpectrum(const GainSpectrum& orig)
        : solver(orig.solver), point(orig.point), reg(orig.reg), T(orig.T), n(orig.n) {}

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
    double getGain(double wavelength);
};

/**
 * Cached luminescence spectrum
 */
template <typename GeometryT> struct PLASK_SOLVER_API LuminescenceSpectrum {
    FermiNewGainSolver<GeometryT>* solver;  ///< Source solver
    Vec<2> point;                           ///< Point in which the luminescence is calculated

    /// Active region containing the point
    size_t reg;

    double T;                   ///< Temperature
    double n;                   ///< Carriers concentration
    unique_ptr<Levels> levels;  ///< Computed energy levels
    std::unique_ptr<kubly::wzmocnienie> gMod;

    LuminescenceSpectrum(FermiNewGainSolver<GeometryT>* solver, const Vec<2> point);

    LuminescenceSpectrum(const LuminescenceSpectrum& orig)
        : solver(orig.solver), point(orig.point), reg(orig.reg), T(orig.T), n(orig.n) {}

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
     * Get luminescence at given wavelength
     * \param wavelength wavelength to get luminescence
     * \return luminescence
     */
    double getLuminescence(double wavelength);
};

}}}  // namespace plask::solvers::FermiNew

#endif
