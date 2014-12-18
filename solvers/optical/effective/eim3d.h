// #ifndef PLASK__MODULE_OPTICAL_EIM_HPP
// #define PLASK__MODULE_OPTICAL_EIM_HPP
//
// #include <limits>
//
// #include <plask/plask.hpp>
//
// #include "rootdigger.h"
// #include "bisection.h"
//
// namespace plask { namespace solvers { namespace effective {
//
// /**
//  * Solver performing calculations in 3D Cartesian space using effective index method
//  */
// struct PLASK_SOLVER_API EffectiveIndex3D: public SolverWithMesh<Geometry3D, RectangularMesh<3>> {
//
//     /// Mode symmetry in horizontal axes
//     enum Symmetry {
//         SYMMETRY_DEFAULT,
//         SYMMETRY_POSITIVE,
//         SYMMETRY_NEGATIVE,
//         SYMMETRY_NONE
//     };
//
//     /// Mode polarization
//     enum Polarization {
//         TE,
//         TM,
//     };
//
//     /// Direction of the possible emission
//     enum Emission {
//         FRONT,
//         BACK
//     };
//
//     struct Field {
//         dcomplex F, B;
//         Field() = default;
//         Field(dcomplex f, dcomplex b): F(f), B(b) {}
//         Field operator*(dcomplex a) const { return Field(F*a, B*a); }
//         Field operator/(dcomplex a) const { return Field(F/a, B/a); }
//         Field operator*=(dcomplex a) { F *= a; B *= a; return *this; }
//         Field operator/=(dcomplex a) { F /= a; B /= a; return *this; }
//     };
//
//     struct Matrix {
//         dcomplex ff, fb, bf, bb;
//         Matrix() = default;
//         Matrix(dcomplex t1, dcomplex t2, dcomplex t3, dcomplex t4): ff(t1), fb(t2), bf(t3), bb(t4) {}
//         static Matrix eye() { return Matrix(1.,0.,0.,1.); }
//         static Matrix diag(dcomplex f, dcomplex b) { return Matrix(f,0.,0.,b); }
//         Matrix operator*(const Matrix& T) {
//             return Matrix( ff*T.ff + fb*T.bf,   ff*T.fb + fb*T.bb,
//                            bf*T.ff + bb*T.bf,   bf*T.fb + bb*T.bb );
//         }
//         Field solve(const Field& v) {
//             return Field(bb*v.F - fb*v.B, -bf*v.F + ff*v.B) / (ff*bb - fb*bf);
//         }
//     };
//
//     /// Details of the computed mode
//     struct Mode {
//         EffectiveIndex3D* solver;       ///< Solver this mode belongs to
//         Symmetry symmetry;              ///< Horizontal symmetry of the modes
//         dcomplex neff;                  ///< Stored mode effective index
//         bool have_fields;               ///< Did we compute fields for current state?
//         std::vector<Field,aligned_allocator<Field>> xfields; ///< Computed horizontal fields
//         std::vector<double,aligned_allocator<double>> xweights; ///< Computed horizontal weights
//         double power;                   ///< Mode power [mW]
//
//         Mode(EffectiveIndex3D* solver, Symmetry sym):
//             solver(solver), have_fields(false), xfields(solver->xend), xweights(solver->xend), power(1e-9) {
//             setSymmetry(sym);
//         }
//
//         void setSymmetry(Symmetry sym) {
//             if (solver->geometry->isSymmetric(Geometry::DIRECTION_TRAN)) {
//                 if (sym == SYMMETRY_DEFAULT)
//                     sym = SYMMETRY_POSITIVE;
//                 else if (sym == SYMMETRY_NONE)
//                     throw BadInput(solver->getId(), "For symmetric geometry specify positive or negative symmetry");
//             } else {
//                 if (sym == SYMMETRY_DEFAULT)
//                     sym = SYMMETRY_NONE;
//                 else if (sym != SYMMETRY_NONE)
//                     throw BadInput(solver->getId(), "For non-symmetric geometry no symmetry may be specified");
//             }
//             symmetry = sym;
//         }
//
//         bool operator==(const Mode& other) const {
//             return symmetry == other.symmetry && is_zero( neff - other.neff );
//         }
//     };
//
//   protected:
//
//     friend struct RootDigger;
//
//     size_t xbegin,  ///< First element of horizontal mesh to consider
//            xend,    ///< Last element of horizontal mesh to consider
//            ybegin,  ///< First element of vertical mesh to consider
//            yend;    ///< Last element of vertical mesh to consider
//
//     /// Logger for determinant
//     Data2DLog<dcomplex,dcomplex> log_value;
//
//     /// Cached refractive indices
//     std::vector<std::vector<dcomplex,aligned_allocator<dcomplex>>> nrCache;
//
//     /// Computed horizontal and vertical fields
//     std::vector<Field,aligned_allocator<Field>> yfields;
//
//     /// Vertical field confinement weights
//     std::vector<double,aligned_allocator<double>> yweights;
//
//     /// Computed effective epsilons for each stripe
//     std::vector<dcomplex,aligned_allocator<dcomplex>> epsilons;
//
//     double stripex;             ///< Transverse position of the main stripe
//     double stripez;             ///< Longitudinal position of the main stripe
//
//     Polarization polarization;  ///< Chosen light polarization
//
//     bool recompute_neffs;       ///< Should stripe indices be recomputed
//
//   public:
//
//     Emission emission;          ///< Direction of laser emission
//
//     dcomplex vneff;             ///< Vertical effective index of the main stripe
//
//     double outdist;             ///< Distance outside outer borders where material is sampled
//
//     /// Parameters for longitudinal rootdigger
//     RootDigger::Params root_long;
//
//     /// Parameters for transverse rootdigger
//     RootDigger::Params root_tran;
//
//     /// Parameters for vertical rootdigger
//     RootDigger::Params root_vert;
//
//     /// Computed modes
//     std::vector<Mode> modes;
//
//     /// Receiver for the temperature
//     ReceiverFor<Temperature, Geometry3D> inTemperature;
//
//     /// Receiver for the gain
//     ReceiverFor<Gain, Geometry3D> inGain;
//
//     /// Provider for the computed wavelenght
//     typename ProviderFor<Wavelength>::Delegate outWavelength;
//
//     /// Provider for the computed loss
//     typename ProviderFor<ModalLoss>::Delegate outLoss;
//
//     /// Provider of the optical field
//     typename ProviderFor<LightMagnitude, Geometry3D>::Delegate outLightMagnitude;
//
//     /// Provider of the refractive index
//     typename ProviderFor<RefractiveIndex, Geometry3D>::Delegate outRefractiveIndex;
//
//     /// Provider of the heat absorbed/generated by the light
//     typename ProviderFor<Heat, Geometry3D>::Delegate outHeat;
//
//     EffectiveIndex3D(const std::string& name="");
//
//     virtual ~EffectiveIndex3D() {
//         inTemperature.changedDisconnectMethod(this, &EffectiveIndex3D::onInputChange);
//         inGain.changedDisconnectMethod(this, &EffectiveIndex3D::onInputChange);
//     }
//
//     virtual std::string getClassName() const { return "optical.EffectiveIndex3D"; }
//
//     virtual std::string getClassDescription() const {
//         return "Calculate optical modes and optical field distribution using the effective index method "
//                "in Cartesian three-dimensional space.";
//     }
//
//     virtual void loadConfiguration(plask::XMLReader& reader, plask::Manager& manager);
//
//     /// \return transverse position of the main stripe
//     double getStripeX() const { return stripex; }
//
//     /**
//      * Set position of the main stripe
//      * \param x transverse position of the main stripe
//      */
//     void setStripeX(double x) {
//         stripex = x;
//         invalidate();
//     }
//
//     /// \return longitudinal position of the main stripe
//     double getStripeZ() const { return stripex; }
//
//     /**
//      * Set position of the main stripe
//      * \param z longitudinal position of the main stripe
//      */
//     void setStripeZ(double z) {
//         stripez = z;
//         invalidate();
//     }
//
//     /// \return current polarization
//     Polarization getPolarization() const { return polarization; }
//
//     /**
//      * Set new polarization
//      * \param polar new polarization
//      */
//     void setPolarization(Polarization polar) {
//         polarization = polar;
//         invalidate();
//     }
//
//     /**
//      * Set the simple mesh based on the geometry bounding boxes.
//      **/
//     void setSimpleMesh() {
//         writelog(LOG_INFO, "Creating simple mesh");
//         setMesh(make_shared<RectilinearMesh3DSimpleGenerator>());
//     }
//
//     /**
//      * Find the mode around the specified effective index.
//      * \param wavelenth initial wavelength to search the mode around
//      * \param symmetry_tran transverse mode symmetry
//      * \param symmetry_long longitudinal mode symmetry
//      * \return index of found mode
//      */
//     size_t findMode(dcomplex wavelenth,
//                     Symmetry symmetry_tran=SYMMETRY_DEFAULT, Symmetry symmetry_long=SYMMETRY_DEFAULT);
//
//     /**
//      * Find the modes within the specified range
//      * \param wavelenth1 one corner of the range to browse
//      * \param wavelenth2 another corner of the range to browse
//      * \param symmetry_tran transverse mode symmetry
//      * \param symmetry_long longitudinal mode symmetry
//      * \param resteps minimum number of steps to check function value on real contour
//      * \param imsteps minimum number of steps to check function value on imaginary contour
//      * \param eps approximate error for integrals
//      * \return vector of indices of found modes
//      */
//     std::vector<size_t> findModes(dcomplex wavelenth1=0., dcomplex wavelenth2=0.,
//                                   Symmetry symmetry_tran=SYMMETRY_DEFAULT, Symmetry symmetry_long=SYMMETRY_DEFAULT,
//                                   size_t resteps=256, size_t imsteps=64, dcomplex eps=dcomplex(1e-6,1e-9));
//
//     /**
//      * Compute modal determinant for the whole matrix
//      * \param wavelenth wavelenth to use
//      * \param symmetry mode symmetry
//      */
//     dcomplex getDeterminant(dcomplex wavelenth, Symmetry sym=SYMMETRY_DEFAULT) {
// //        stageOne();
// //        Mode mode(this,sym);
// //        dcomplex det = detS(wavelenth, mode);
// //        return det;
//     }
//
//     /**
//      * Compute field weights
//      * \param num mode number to consider
//      */
//     double getTotalAbsorption(size_t num);
//
//   protected:
//
//     /// Slot called when gain has changed
//     void onInputChange(ReceiverBase&, ReceiverBase::ChangeReason) {
//         invalidate();
//     }
//
// //    /// Initialize the solver
// //    virtual void onInitialize();
//
// //    /// Invalidate the data
// //    virtual void onInvalidate();
//
// //    /// Do we need to have gain
// //    bool need_gain;
//
// //    /**
// //     * Update refractive index cache
// //     * \return \c true if the chache has been changed
// //     */
// //    void updateCache();
//
// //    /**
// //     * Fist stage of computations
// //     * Perform vertical computations
// //     */
// //    void stageOne();
//
// //    /**
// //     * Second stage of computations
// //     * Perform transverse computations
// //     */
// //    void stageTwo();
//
// //    /**
// //     * Compute field weights basing on solution for given stripe. Also compute data for determining vertical fields
// //     * \param stripe main stripe number
// //     */
// //    void computeWeights(size_t stripe);
//
// //    /**
// //     * Normalize horizontal fields, so multiplying LightMagnitude by power gives proper LightMagnitude in (V/m)²
// //     * \param kx computed horizontal propagation constants
// //     */
// //    void normalizeFields(Mode& mode, const std::vector<dcomplex,aligned_allocator<dcomplex>>& kx);
//
// //    /**
// //     * Compute field weights
// //     * \param mode mode to consider
// //     */
// //    double getTotalAbsorption(const Mode& mode);
//
// //    /**
// //     * Compute S matrix determinant for one stripe
// //     * \param x vertical effective index
// //     * \param NR refractive indices
// //     * \param save if \c true, the fields are saved to yfields
// //     */
// //    dcomplex detS1(const dcomplex& x, const std::vector<dcomplex,aligned_allocator<dcomplex>>& NR, bool save=false);
//
// //    /**
// //     * Return S matrix determinant for the whole structure
// //     * \param x effective index
// //     * \param save if \c true, the fields are saved to xfields
// //     */
// //    dcomplex detS(const dcomplex& x, Mode& mode, bool save=false);
//
// //    /// Insert mode to the list or return the index of the exiting one
// //    size_t insertMode(const Mode& mode) {
// //        for (size_t i = 0; i != modes.size(); ++i)
// //            if (modes[i] == mode) return i;
// //        modes.push_back(mode);
// //        outNeff.fireChanged();
// //        outLightMagnitude.fireChanged();
// //        return modes.size()-1;
// //    }
//
// //    /// Return number of found modes
// //    size_t nmodes() const {
// //        return modes.size();
// //    }
//
// //    /**
// //     * Return mode effective index
// //     * \param n mode number
// //     */
// //    dcomplex getEffectiveIndex(size_t n) {
// //        if (n >= modes.size()) throw NoValue(EffectiveIndex::NAME);
// //        return modes[n].neff;
// //    }
//
// //    struct LightMagnitudeDataBase;
// //    struct LightMagnitudeDataInefficient;
// //    struct LightMagnitudeDataEfficient;
// //    struct HeatDataImpl;
//
// //    /// Method computing the distribution of light intensity
// //    const LazyData<double> getLightMagnitude(int num, shared_ptr<const plask::MeshD<2>> dst_mesh, plask::InterpolationMethod=INTERPOLATION_DEFAULT);
//
// //    /// Get used refractive index
// //    const LazyData<Tensor3<dcomplex>> getRefractiveIndex(shared_ptr<const MeshD<2> > dst_mesh, InterpolationMethod=INTERPOLATION_DEFAULT);
//
// //    /// Get generated/absorbed heat
// //    const LazyData<double> getHeat(shared_ptr<const MeshD<2> > dst_mesh, InterpolationMethod method=INTERPOLATION_DEFAULT);
//
// };
//
//
// }}} // namespace plask::solvers::effective
//
// #endif // PLASK__MODULE_OPTICAL_EIM_HPP
