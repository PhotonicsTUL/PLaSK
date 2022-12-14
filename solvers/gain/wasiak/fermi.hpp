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
 * \file
 * Sample solver header for your solver
 */
#ifndef PLASK__SOLVER_GAIN_FERMI_H
#define PLASK__SOLVER_GAIN_FERMI_H

#include <plask/plask.hpp>
#include "gainQW.hpp"

namespace plask { namespace solvers { namespace fermi {

template <typename GeometryT> struct GainSpectrum;

/**
 * Gain solver using Fermi Golden Rule
 */
template <typename GeometryType>
struct PLASK_SOLVER_API FermiGainSolver: public SolverWithMesh<GeometryType, MeshAxis>
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
            throw plask::Exception("FermiGainSolver requires solid layers.");
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

        shared_ptr<Material> materialQW;        ///< Quantum well material
        shared_ptr<Material> materialBarrier;   ///< Barrier material
        double qwlen;                           ///< Single quantum well thickness [??]
        double qwtotallen;                      ///< Total quantum wells thickness [??]
        double totallen;                        ///< Total active region thickness [??]

        /**
         * Summarize active region, check for appropriateness and compute some values
         * \param solver solver
         */
        void summarize(const FermiGainSolver<GeometryType>* solver) {
            auto bbox = layers->getBoundingBox();
            totallen = 1e4 * (bbox.upper[1] - bbox.lower[1]);  // 1e4: ??m -> ??
            size_t qwn = 0;
            qwtotallen = 0.;
            bool lastbarrier = true;
            for (const auto& layer: layers->children) {
                auto block = static_cast<Block<2>*>(static_cast<Translation<2>*>(layer.get())->getChild().get());
                auto material = block->singleMaterial();
                if (!material) throw plask::Exception("FermiGainSolver requires solid layers.");
                if (static_cast<Translation<2>*>(layer.get())->getChild()->hasRole("QW")) {
                    if (!materialQW)
                        materialQW = material;
                    else if (*material != *materialQW)
                        throw Exception("{0}: Multiple quantum well materials in active region.", solver->getId());
                    auto bbox = static_cast<GeometryObjectD<2>*>(layer.get())->getBoundingBox();
                    qwtotallen += bbox.upper[1] - bbox.lower[1];
                    if (lastbarrier) ++qwn;
                    else solver->writelog(LOG_WARNING, "Considering two adjacent quantum wells as one");
                    lastbarrier = false;
                } else {
                    if (!materialBarrier)
                        materialBarrier = material;
                    else if (!is_zero(material->Me(300).c00 - materialBarrier->Me(300).c00) ||
                             !is_zero(material->Me(300).c11 - materialBarrier->Me(300).c11) ||
                             !is_zero(material->Mhh(300).c00 - materialBarrier->Mhh(300).c00) ||
                             !is_zero(material->Mhh(300).c11 - materialBarrier->Mhh(300).c11) ||
                             !is_zero(material->Mlh(300).c00 - materialBarrier->Mlh(300).c00) ||
                             !is_zero(material->Mlh(300).c11 - materialBarrier->Mlh(300).c11) ||
                             !is_zero(material->CB(300) - materialBarrier->CB(300)) ||
                             !is_zero(material->VB(300) - materialBarrier->VB(300)))
                        throw Exception("{0}: Multiple barrier materials around active region.", solver->getId());
                    lastbarrier = true;
                }
            }
            qwtotallen *= 1e4; // ??m -> ??
            qwlen = qwtotallen / qwn;
        }
    };

    shared_ptr<Material> materialSubstrate;   ///< Substrate material

    ///< List of active regions
    std::vector<ActiveRegionInfo> regions;

    ///< Optional externally set energy levels
    boost::optional<QW::ExternalLevels> extern_levels;

    /// Receiver for temperature.
    ReceiverFor<Temperature,GeometryType> inTemperature;

    /// Receiver for carriers concentration in the active region
    ReceiverFor<CarriersConcentration,GeometryType> inCarriersConcentration;

    /// Provider for gain distribution
    typename ProviderFor<Gain,GeometryType>::Delegate outGain;

    FermiGainSolver(const std::string& name="");

    virtual ~FermiGainSolver();

    virtual std::string getClassName() const;

    virtual void loadConfiguration(plask::XMLReader& reader, plask::Manager& manager);

    /// Function computing energy levels
    std::deque<std::tuple<std::vector<double>,std::vector<double>,std::vector<double>,double,double>>
    determineLevels(double T, double n);

  protected:

    friend struct GainSpectrum<GeometryType>;
    friend class QW::gain;

    double cond_waveguide_depth;    ///< waveguide conduction band depth [eV]
    double vale_waveguide_depth;    ///< waveguide valence band depth [eV]
    double cond_qw_shift;           ///< additional conduction band shift for qw [eV]
    double vale_qw_shift;           ///< additional valence band shift for qw [eV]
    double lifetime;                ///< stimulated emission lifetime [ps]
    double matrixelem;              ///< optical matrix element [m0*eV]
    double differenceQuotient;      ///< difference quotient of dG_dn derivative

    QW::gain getGainModule(double wavelength, double T, double n, const ActiveRegionInfo& region);

    void prepareLevels(QW::gain& gmodule, const ActiveRegionInfo& region) {
        if (extern_levels) {
            gmodule.przygobl_n(*extern_levels, gmodule.przel_dlug_z_angstr(region.qwtotallen)); // earlier: qwtotallen
        } else {
            gmodule.przygobl_n(gmodule.przel_dlug_z_angstr(region.qwlen)); // earlier: qwtotallen
        }
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
    }

    /**
     * Detect active regions.
     * Store information about them in the \p regions field.
     */
    void detectActiveRegions();

    struct DataBase;
    struct GainData;
    struct DgdnData;

    /**
     * Compute the gain on the mesh. This method is called by gain provider.
     * \param what what to return (gain or its carriera derivative)
     * \param dst_mesh destination mesh
     * \param wavelength wavelength to compute gain for
     * \param interp interpolation method
     * \return gain distribution
     */
    const LazyData<Tensor2<double>> getGain(Gain::EnumType what, const shared_ptr<const MeshD<2>>& dst_mesh, double wavelength, InterpolationMethod interp=INTERPOLATION_DEFAULT);

  public:

    bool if_strain;                 ///< Consider strain in QW?

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
                solver->inTemperature.changedConnectMethod(this, &GainSpectrum::onTChange);
                solver->inCarriersConcentration.changedConnectMethod(this, &GainSpectrum::onNChange);
                return;
            };
        }
        throw BadInput(solver->getId(), "Point {0} does not belong to any active region", point);
    }

    void onTChange(ReceiverBase&, ReceiverBase::ChangeReason) { T = NAN; }

    void onNChange(ReceiverBase&, ReceiverBase::ChangeReason) { n = NAN; }

    ~GainSpectrum() {
        solver->inTemperature.changedDisconnectMethod(this, &GainSpectrum::onTChange);
        solver->inCarriersConcentration.changedDisconnectMethod(this, &GainSpectrum::onNChange);
    }

    /**
     * Get gain at given wavelength
     * \param wavelength wavelength to get gain
     * \return gain
     */
    double getGain(double wavelength) {
        #pragma omp critical
        {
            if (isnan(T)) T = solver->inTemperature(plask::make_shared<const OnePointMesh<2>>(point))[0];
            if (isnan(n)) n = solver->inCarriersConcentration(plask::make_shared<const OnePointMesh<2>>(point))[0];
        }
        return solver->getGainModule(wavelength, T, n, *region) // returns gain for single QW layer!
            .Get_gain_at_n(solver->nm_to_eV(wavelength), region->qwlen); // earlier: qwtotallen
    }
};




}}} // namespace

#endif

