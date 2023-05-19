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
#ifndef PLASK__MODULE_OPTICAL_EFM_HPP
#define PLASK__MODULE_OPTICAL_EFM_HPP

#include <limits>

#include <plask/plask.hpp>
#include <camos/camos.h>

#include "rootdigger.hpp"
#include "bisection.hpp"

namespace plask { namespace optical { namespace effective {

static constexpr int MH = 2; // Hankel function type (1 or 2)

/**
 * Solver performing calculations in 2D Cartesian space using effective index method
 */
struct PLASK_SOLVER_API EffectiveFrequencyCyl: public SolverWithMesh<Geometry2DCylindrical, RectangularMesh<2>> {

    struct FieldZ {
        dcomplex F, B;
        FieldZ() = default;
        FieldZ(dcomplex f, dcomplex b): F(f), B(b) {}
        FieldZ operator*(dcomplex a) const { return FieldZ(F*a, B*a); }
        FieldZ operator/(dcomplex a) const { return FieldZ(F/a, B/a); }
        FieldZ operator*=(dcomplex a) { F *= a; B *= a; return *this; }
        FieldZ operator/=(dcomplex a) { F /= a; B /= a; return *this; }
    };

    struct MatrixZ {
        dcomplex ff, fb, bf, bb;
        MatrixZ() = default;
        MatrixZ(dcomplex t1, dcomplex t2, dcomplex t3, dcomplex t4): ff(t1), fb(t2), bf(t3), bb(t4) {}
        static MatrixZ eye() { return MatrixZ(1.,0.,0.,1.); }
        static MatrixZ diag(dcomplex f, dcomplex b) { return MatrixZ(f,0.,0.,b); }
        MatrixZ operator*(const MatrixZ& T) {
            return MatrixZ(ff*T.ff + fb*T.bf,   ff*T.fb + fb*T.bb,
                           bf*T.ff + bb*T.bf,   bf*T.fb + bb*T.bb);
        }
        FieldZ operator*(const FieldZ& v) {
            return FieldZ(ff*v.F + fb*v.B, bf*v.F + bb*v.B);
        }
        FieldZ solve(const FieldZ& v) {
            return FieldZ(bb*v.F - fb*v.B, -bf*v.F + ff*v.B) / (ff*bb - fb*bf);
        }
    };

    struct FieldR {
        dcomplex J, H;
        FieldR() = default;
        FieldR(dcomplex j, dcomplex h): J(j), H(h) {}
        FieldR operator*(dcomplex a) const { return FieldR(a*J, a*H); }
        FieldR operator/(dcomplex a) const { return FieldR(J/a, H/a); }
        FieldR& operator*=(dcomplex a) { J *= a; H *= a; return *this; }
        FieldR& operator/=(dcomplex a) { J /= a; H /= a; return *this; }
    };

    struct MatrixR {
        dcomplex JJ, JH, HJ, HH;
        MatrixR(dcomplex jj, dcomplex jh, dcomplex hj, dcomplex hh): JJ(jj), JH(jh), HJ(hj), HH(hh) {}
        static MatrixR eye() { return MatrixR(1.,0.,0.,1.); }
        static MatrixR diag(dcomplex j, dcomplex h) { return MatrixR(j,0.,0.,h); }
        MatrixR operator*(dcomplex c) { return MatrixR(c * JJ, c * JH, c * HJ, c * HH); }
        friend MatrixR operator*(dcomplex c, const MatrixR& M) { return MatrixR(c * M.JJ, c * M.JH, c * M.HJ, c * M.HH); }
        MatrixR operator/(dcomplex d) { dcomplex c = 1./d; return MatrixR(c * JJ, c * JH, c * HJ, c * HH); }
        MatrixR& operator*=(dcomplex c) { JJ *= c; JH *= c; HJ *= c; HH *= c; return *this; }
        MatrixR& operator/=(dcomplex d) { dcomplex c = 1./d; JJ *= c; JH *= c; HJ *= c; HH *= c; return *this; }
        FieldR operator*(const FieldR& v) {
            return FieldR(JJ*v.J + JH*v.H, HJ*v.J + HH*v.H);
        }
        MatrixR operator*(const MatrixR& o) {
            return MatrixR(JJ*o.JJ + JH*o.HJ, JJ*o.JH + JH*o.HH,
                           HJ*o.JJ + HH*o.HJ, HJ*o.JH + HH*o.HH);
        }
        FieldR solve(const FieldR& v) {
            return FieldR(HH*v.J - JH*v.H, -HJ*v.J + JJ*v.H) / (JJ*HH - JH*HJ);
        }
        MatrixR solve(const MatrixR& o) {
            MatrixR result(HH*o.JJ - JH*o.HJ, HH*o.JH - JH*o.HH, -HJ*o.JJ + JJ*o.HJ, -HJ*o.JH + JJ*o.HH);
            result /= (JJ*HH - JH*HJ);
            return result;
        }
    };


    /// Direction of the possible emission
    enum Emission {
        TOP,        ///< Top emission
        BOTTOM      ///< Bottom emission
    };

    /// Radial determinant modes
    enum Determinant {
        DETERMINANT_INWARDS,     ///< Use out->in transfer matrix method
        DETERMINANT_OUTWARDS,    ///< Use out->in transfer matrix method
        DETERMINANT_FULL        ///< Construct one matrix for all layers
    };

    /// Details of the computed mode
    struct Mode {
        EffectiveFrequencyCyl* solver;      ///< Solver this mode belongs to
        int m;                              ///< Number of the LP_mn mode describing angular dependence
        bool have_fields;                   ///< Did we compute fields for current state?
        std::vector<FieldR,aligned_allocator<FieldR>> rfields; ///< Computed horizontal fields
        std::vector<double,aligned_allocator<double>> rweights; /// Computed normalized lateral field integral for each stripe
        dcomplex lam;                       ///< Stored wavelength
        double power;                       ///< Mode power [mW]

        Mode(EffectiveFrequencyCyl* solver, int m=0):
            solver(solver), m(m), have_fields(false), rfields(solver->rsize), rweights(solver->rsize), power(1.) {}

        bool operator==(const Mode& other) const {
            return m == other.m && is_zero(lam - other.lam);
        }

        /// Compute horizontal part of the field
        dcomplex rField(double r) const {
            double Jr, Ji, Hr, Hi;
            long nz, ierr;
            size_t ir = solver->mesh->axis[0]->findIndex(r); if (ir > 0) --ir; if (ir >= solver->veffs.size()) ir = solver->veffs.size()-1;
            dcomplex x = r * solver->k0 * sqrt(solver->nng[ir] * (solver->veffs[ir] - solver->freqv(lam)));
            if (real(x) < 0.) x = -x;
            if (imag(x) > SMALL) x = -x;
            if (ir == solver->rsize-1) {
                Jr = Ji = 0.;
            } else {
                zbesj(x.real(), x.imag(), m, 1, 1, &Jr, &Ji, nz, ierr);
                if (ierr != 0)
                    throw ComputationError(solver->getId(), "Could not compute J({0}, {1}) @ r = {2}um", m, str(x), r);
            }
            if (ir == 0) {
                Hr = Hi = 0.;
            } else {
                zbesh(x.real(), x.imag(), m, 1, MH, 1, &Hr, &Hi, nz, ierr);
                if (ierr != 0)
                    throw ComputationError(solver->getId(), "Could not compute H({0}, {1}) @ r = {2}um", m, str(x), r);
            }
            return rfields[ir].J * dcomplex(Jr, Ji) + rfields[ir].H * dcomplex(Hr, Hi);
        }

        /// Return mode loss
        double loss() const {
            return imag(4e7*PI / lam);
        }
    };

    /// Convert wavelength to the frequency parameter
    dcomplex freqv(dcomplex lam) {
        return 2. - 4e3*PI / lam / k0;
    }

    /// Convert frequency parameter to the wavelength
    dcomplex lambda(dcomplex freq) {
        return 2e3*PI / (k0 * (1. - freq/2.));
    }

  protected:

    friend struct RootDigger;

    /// Logger for char_val
    DataLog<dcomplex,dcomplex> log_value;

    size_t rsize,   ///< Last element of horizontal mesh to consider
           zbegin,  ///< First element of vertical mesh to consider
           zsize;   ///< Last element of vertical mesh to consider

    /// Cached refractive indices
    std::vector<std::vector<dcomplex,aligned_allocator<dcomplex>>> nrCache;

    /// Cached group indices
    std::vector<std::vector<dcomplex,aligned_allocator<dcomplex>>> ngCache;

    /// Computed vertical fields
    std::vector<FieldZ> zfields;

    /// Vertical field confinement weights
    std::vector<double,aligned_allocator<double>> zintegrals;

    /// Computed effective frequencies for each stripe
    std::vector<dcomplex,aligned_allocator<dcomplex>> veffs;

    /// Computed weighted indices for each stripe
    std::vector<dcomplex,aligned_allocator<dcomplex>> nng;

    /// Old value of k0 to detect changes
    dcomplex old_k0;

    /// Direction of laser emission
    Emission emission;

    /// Slot called when gain has changed
    void onInputChange(ReceiverBase&, ReceiverBase::ChangeReason) {
        cache_outdated = true;
    }

    /**
     * Stripe number to use for vertical computations.
     * -1 means to compute all stripes as in the proper EFM
     */
    int rstripe;

  public:

    /// Radial determinant mode
    Determinant determinant;

    /// Return the main stripe number
    int getStripe() const {
        return rstripe;
    }

    /// Set stripe for computations
    void setStripe(int stripe) {
        if (!mesh) setSimpleMesh();
        if (stripe < 0 || std::size_t(stripe) >= mesh->axis[0]->size())
            throw BadInput(getId(), "Wrong stripe number specified");
        rstripe = stripe;
        invalidate();
    }

    /// Get position of the main stripe
    double getStripeR() const {
        if (rstripe == -1 || !mesh) return NAN;
        return mesh->axis[0]->at(rstripe);
    }

    /**
     * Set position of the main stripe
     * \param r horizontal position of the main stripe
     */
    void setStripeR(double r=0.) {
        if (!mesh) setSimpleMesh();
        if (r < 0) throw BadInput(getId(), "Radial position cannot be negative");
        rstripe = int(std::lower_bound(mesh->axis[0]->begin()+1, mesh->axis[0]->end(), r) - mesh->axis[0]->begin() - 1);
        invalidate();
    }

    /// Use all stripes
    void useAllStripes() {
        rstripe = -1;
        invalidate();
    }

    /**
     * Return radial effective index part at specified position
     * \param horizontal position
     */
    dcomplex getDeltaNeff(double r) {
        stageOne();
        if (r < 0) throw BadInput(getId(), "Radial position cannot be negative");
        size_t ir = mesh->axis[0]->findIndex(r); if (ir > 0) --ir; if (ir >= veffs.size()) ir = veffs.size()-1;
        return sqrt(nng[ir] * veffs[ir]);
    }

    /**
     * Return effective index at specified position
     * \param horizontal position
     */
    dcomplex getNNg(double r) {
        stageOne();
        if (r < 0) throw BadInput(getId(), "Radial position cannot be negative");
        size_t ir = mesh->axis[0]->findIndex(r); if (ir > 0) --ir; if (ir >= veffs.size()) ir = veffs.size()-1;
        return sqrt(nng[ir]);
    }

    // Parameters for rootdigger
    RootDigger::Params root;        ///< Parameters for horizontal root digger
    RootDigger::Params stripe_root; ///< Parameters for vertical root diggers

    /// Allowed relative power integral precision
    double perr;

    /// Current value of reference normalized frequency [1/µm]
    dcomplex k0;

    /// 'Vertical wavelength' used as a helper for searching vertical modes
    dcomplex vlam;

    /// Computed modes
    std::vector<Mode> modes;

    /// Receiver for the temperature
    ReceiverFor<Temperature, Geometry2DCylindrical> inTemperature;

    /// Receiver for the gain
    ReceiverFor<Gain, Geometry2DCylindrical> inGain;

    /// Receiver for the carriers concentration
    ReceiverFor<CarriersConcentration, Geometry2DCylindrical> inCarriersConcentration;

    /// Provider for computed resonant wavelength
    typename ProviderFor<ModeWavelength>::Delegate outWavelength;

    /// Provider for computed modal extinction
    typename ProviderFor<ModeLoss>::Delegate outLoss;

    /// Provider of optical field
    typename ProviderFor<ModeLightMagnitude, Geometry2DCylindrical>::Delegate outLightMagnitude;

    /// Provider of optical field
    typename ProviderFor<ModeLightE, Geometry2DCylindrical>::Delegate outLightE;

    /// Provider of refractive index
    typename ProviderFor<RefractiveIndex, Geometry2DCylindrical>::Delegate outRefractiveIndex;

    /// Provider of the heat absorbed/generated by the light
    typename ProviderFor<Heat, Geometry2DCylindrical>::Delegate outHeat;

    EffectiveFrequencyCyl(const std::string& name="");

    virtual ~EffectiveFrequencyCyl() {
        inTemperature.changedDisconnectMethod(this, &EffectiveFrequencyCyl::onInputChange);
        inGain.changedDisconnectMethod(this, &EffectiveFrequencyCyl::onInputChange);
        inCarriersConcentration.changedDisconnectMethod(this, &EffectiveFrequencyCyl::onInputChange);
    }

    std::string getClassName() const override { return "optical.EffectiveFrequencyCyl"; }

    std::string getClassDescription() const override {
        return "Calculate optical modes and optical field distribution using the effective index method "
               "in Cartesian two-dimensional space.";
    }

    void loadConfiguration(plask::XMLReader& reader, plask::Manager& manager) override;

    /// Get emission direction
    ///\return emission direction
    Emission getEmission() const { return emission; }

    /// Set emission direction
    /// \param emis new emissjon direction
    void setEmission(Emission emis) {
        emission = emis;
        for (auto& mode: modes)
            mode.have_fields = false;
    }

    /**
     * Set the simple mesh based on the geometry bounding boxes.
     **/
    void setSimpleMesh() {
        writelog(LOG_DETAIL, "Creating simple mesh");
        setMesh(plask::make_shared<RectangularMesh2DSimpleGenerator>());
    }

    /**
     * Set up the horizontal mesh. Horizontal division is provided while vertical one is created basing on the geometry bounding boxes.
     *
     * \param meshx horizontal mesh
     **/
    void setHorizontalMesh(shared_ptr<MeshAxis> meshx) {
        writelog(LOG_DETAIL, "Setting horizontal mesh");
        if (!geometry) throw NoChildException();
        auto meshxy = makeGeometryGrid(geometry->getChild());
        meshxy->setTran(meshx);
        setMesh(meshxy);
    }

    /// Get asymptotic flag
    bool getAsymptotic() const { return asymptotic; }

    /// Set asymptotic flag
    void setAsymptotic(bool value) {
        asymptotic = value;
        invalidate();
    }

    /**
     * Find the mode around the specified effective wavelength.
     *
     * This method remembers the determined mode, for retrieval of the field profiles.
     *
     * \param lambda initial wavelength close to the solution
     * \param m number of the LP_mn mode describing angular dependence
     * \return index of the found mode
     */
    size_t findMode(dcomplex lambda, int m=0);

    /**
     * Find the modes within the specified range
     *
     * This method \b does \b not remember the determined modes!
     *
     * \param lambda1 one corner of the range to browse
     * \param lambda2 another corner of the range to browse
     * \param m number of the LP_mn mode describing angular dependence
     * \param resteps minimum number of steps to check function value on real contour
     * \param imsteps minimum number of steps to check function value on imaginary contour
     * \param eps approximate error for integrals
     * \return vector of indices of determined modes
     */
    std::vector<size_t> findModes(plask::dcomplex lambda1=0., plask::dcomplex lambda2=0., int m=0, size_t resteps=256, size_t imsteps=64, dcomplex eps=dcomplex(1e-6,1e-9));

    /**
     * Compute vectical modal determinant
     * \param vlambda vertical plane-wave wavelength
     */
    dcomplex getVertDeterminant(dcomplex vlambda) {
        updateCache();
        if (rstripe < 0) throw BadInput(getId(), "This works only for the weighted approach");
        if (vlam == 0. && isnan(k0.real())) throw BadInput(getId(), "No reference wavelength `lam0` specified");
        dcomplex v = freqv(vlambda);
        return this->detS1(v, nrCache[rstripe], ngCache[rstripe]);
    }

    /**
     * Compute modal determinant for the whole matrix
     * \param lambda wavelength
     * \param m number of the LP_mn mode describing angular dependence
     */
    dcomplex getDeterminant(dcomplex lambda, int m=0) {
    if (isnan(k0.real())) throw BadInput(getId(), "No reference wavelength `lam0` specified");
        stageOne();
        Mode mode(this,m);
        dcomplex det = detS(lambda, mode);
        // log_value(v, det);
        return det;
    }

    /**
     * Set particular value of the effective wavelength, e.g. to one of the values returned by findModes.
     * If it is not proper mode, exception is throw.
     * \param clambda complex wavelength of the mode
     * \return index of the set mode
     */
    size_t setMode(dcomplex clambda, int m=0);

    /**
     * Set particular value of the effective wavelength, e.g. to one of the values returned by findModes.
     * If it is not proper mode, exception is throw.
     * \param lambda wavelength of the mode
     * \param loss modal loss (as returned by outLoss)
     * \param m number of the LP_mn mode describing angular dependence
     * \return index of the set mode
     */
    inline size_t setMode(double lambda, double loss, int m=0) {
        return setMode(dcomplex(lambda, -lambda*lambda / (4e7*PI) * loss), m);
    }

    /// Clear computed modes
    void clearModes() {
        modes.clear();
    }

    /**
     * Return total amount of energy absorbed by the matter in a unit time.
     * \param mode mode to analyze
     */
    double getTotalAbsorption(Mode& mode);

    /**
     * Return total amount of energy absorbed by the matter in a unit time.
     * \param num mode number
     */
    double getTotalAbsorption(size_t num);

    /**
     * Return total amount of energy generated in the gain region in a unit time.
     * \param mode mode to analyze
     */
    double getGainIntegral(Mode& mode);

    /**
     * Return total amount of energy absorbed by the matter in a unit time.
     * \param num mode number
     */
    double getGainIntegral(size_t num);

  protected:

    /// Do we need to compute gain
    bool need_gain;

    /// Indicator that we need to recompute the effective indices
    bool cache_outdated;

    /// Indicator if we have veffs foe the current cache
    bool have_veffs;

    /// Indicator if we want an asymptotic lateral solution in the outermost layer
    bool asymptotic;

    /// Initialize the solver
    void onInitialize() override;

    /// Invalidate the data
    void onInvalidate() override;

    /**
     * Update refractive index cache
     */
    void updateCache();

    /**
     * Fist stage of computations
     * Perform vertical computations
     */
    void stageOne();

    /// Return S matrix determinant for one stripe
    dcomplex detS1(const dcomplex& v, const std::vector<dcomplex,aligned_allocator<dcomplex>>& NR,
                   const std::vector<dcomplex,aligned_allocator<dcomplex>>& NG, std::vector<FieldZ>* saveto=nullptr);

    /// Compute stripe averaged n ng
    void computeStripeNNg(size_t stripe, bool save_integrals=false);

    /** Integrate horizontal field
     * \param mode mode to integrate
     */
    double integrateBessel(Mode& mode);

    /// Compute Bessel functions
    void computeBessel(size_t i, dcomplex v, const Mode& mode, dcomplex* J1, dcomplex* H1, dcomplex* J2, dcomplex* H2);

    /// Return S matrix determinant for the whole structure
    dcomplex detS(const plask::dcomplex& lam, Mode& mode, bool save=false);

    /// Obtain main stripe
    size_t getMainStripe() {
        if (rstripe < 0) {
            size_t stripe = 0;
            // Look for the innermost stripe with not constant refractive index
            bool all_the_same = true;
            while (all_the_same) {
                dcomplex same_nr = nrCache[stripe].front();
                dcomplex same_ng = ngCache[stripe].front();
                for (auto nr = nrCache[stripe].begin(), ng = ngCache[stripe].begin(); nr != nrCache[stripe].end(); ++nr, ++ng)
                    if (*nr != same_nr || *ng != same_ng) { all_the_same = false; break; }
                if (all_the_same) ++stripe;
            }
            writelog(LOG_DETAIL, "Vertical field distribution taken from stripe {0}", stripe);
            return stripe;
        } else {
            return rstripe;
        }
    }

    /// Insert mode to the list or return the index of the exiting one
    size_t insertMode(const Mode& mode) {
        for (size_t i = 0; i != modes.size(); ++i)
        if (modes[i] == mode) return i;
        modes.push_back(mode);
        outWavelength.fireChanged();
        outLoss.fireChanged();
        outLightMagnitude.fireChanged();
        outLightE.fireChanged();
        return modes.size()-1;
    }

    /// Return number of found modes
    size_t nmodes() const {
        return modes.size();
    }

    /**
     * Return mode wavelength
     * \param n mode number
     */
    double getWavelength(size_t n) {
        if (n >= modes.size()) throw NoValue(ModeWavelength::NAME);
        return real(modes[n].lam);
    }

    /**
     * Return mode modal loss
     * \param n mode number
     */
    double getModalLoss(size_t n) {
        if (n >= modes.size()) throw NoValue(ModeLoss::NAME);
        return imag(4e7*PI / modes[n].lam);  // 2e4  2/µm -> 2/cm
    }

    template <typename T> struct FieldDataBase;
    template <typename T> struct FieldDataInefficient;
    template <typename T> struct FieldDataEfficient;
    struct HeatDataImpl;

    /// Method computing the distribution of light intensity
    const LazyData<double> getLightMagnitude(std::size_t num, const shared_ptr<const MeshD<2>>& dst_mesh, InterpolationMethod=INTERPOLATION_DEFAULT);

    /// Method computing the distribution of the light electric field
    const LazyData<Vec<3,dcomplex>> getElectricField(std::size_t num, const shared_ptr<const plask::MeshD<2>>& dst_mesh, plask::InterpolationMethod=INTERPOLATION_DEFAULT);

    /// Get used refractive index
    const LazyData<Tensor3<dcomplex>> getRefractiveIndex(const shared_ptr<const MeshD<2> >& dst_mesh, InterpolationMethod=INTERPOLATION_DEFAULT);

    /// Get generated/absorbed heat
    const LazyData<double> getHeat(const shared_ptr<const MeshD<2> > &dst_mesh, InterpolationMethod method=INTERPOLATION_DEFAULT);
};


}}} // namespace plask::optical::effective

#endif // PLASK__MODULE_OPTICAL_EFM_HPP
