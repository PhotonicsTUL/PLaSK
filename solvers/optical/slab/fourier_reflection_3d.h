// #ifndef PLASK__SOLVER__OPTICAL__SLAB_FOURIER_REFLECTION_3D_H
// #define PLASK__SOLVER__OPTICAL__SLAB_FOURIER_REFLECTION_3D_H
// 
// #include <plask/plask.hpp>
// 
// #include "reflection_base.h"
// #include "expansion_pw3d.h"
// 
// namespace plask { namespace solvers { namespace slab {
// 
// /**
//  * Reflection transformation solver in Cartesian 3D geometry.
//  */
// struct FourierReflection3D: public ReflectionSolver<Geometry3DCartesian> {
// 
//     std::string getClassName() const { return "optical.FourierReflection3D"; }
// 
//     struct Mode {
//         FourierReflection3D* solver;                            ///< Solver this mode belongs to
//         ExpansionPW3D::Component symmetry_tran;                 ///< Mode symmetry in tran direction
//         ExpansionPW3D::Component symmetry_long;                 ///< Mode symmetry in long direction
//         dcomplex k0;                                            ///< Stored mode frequency
//         dcomplex klong;                                         ///< Stored mode effective index
//         dcomplex ktran;                                         ///< Stored mode transverse wavevector
//         double power;                                           ///< Mode power [mW]
// 
//         Mode(FourierReflection3D* solver): solver(solver), power(1.) {}
// 
//         bool operator==(const Mode& other) const {
//             return is_zero(k0 - other.k0) && is_zero(klong - other.klong) && is_zero(ktran - other.ktran)
//                 && (!solver->expansion.symmetric || symmetry == other.symmetry)
//                 && (!solver->expansion.separated || polarization == other.polarization)
//             ;
//         }
//     };
// 
//   protected:
// 
//     /// Maximum order of the orthogonal base
//     size_t size;
// 
//     /// Class responsoble for computing expansion coefficients
//     ExpansionPW3D expansion;
// 
//     void onInitialize();
// 
//     void onInvalidate();
// 
//     void computeCoefficients() override {
//         expansion.computeMaterialCoefficients();
//     }
// 
//   public:
// 
//     /// Computed modes
//     std::vector<Mode> modes;
// 
//     /// Mesh multiplier for finer computation of the refractive indices
//     size_t refine;
// 
//     /// Lateral PMLs
//     PML pml;
// 
//     /// Provider for computed effective index
//     ProviderFor<EffectiveIndex>::Delegate outNeff;
// 
//     FourierReflection3D(const std::string& name="");
// 
//     void loadConfiguration(XMLReader& reader, Manager& manager);
// 
//     /**
//      * Find the mode around the specified effective index.
//      * This method remembers the determined mode, for retrieval of the field profiles.
//      * \param neff initial effective index to search the mode around
//      * \return determined effective index
//      */
//     size_t findMode(dcomplex neff);
// 
//     /// Get order of the orthogonal base
//     size_t getSize() const { return size; }
//     /// Set order of the orthogonal base
//     void setSize(size_t n) {
//         size = n;
//         invalidate();
//     }
// 
//     /// Return current mode symmetry
//     ExpansionPW3D::Component getSymmetry() const { return expansion.symmetry; }
//     /// Set new mode symmetry
//     void setSymmetry(ExpansionPW3D::Component symmetry) {
//         if (geometry && !geometry->isSymmetric(Geometry3DCartesian::DIRECTION_TRAN))
//             throw BadInput(getId(), "Symmetry not allowed for asymmetric structure");
//         if (expansion.initialized) {
//             if (expansion.symmetric && symmetry == ExpansionPW3D::E_UNSPECIFIED)
//                 throw Exception("%1%: Cannot remove mode symmetry now -- invalidate the solver first", getId());
//             if (!expansion.symmetric && symmetry != ExpansionPW3D::E_UNSPECIFIED)
//                 throw Exception("%1%: Cannot add mode symmetry now -- invalidate the solver first", getId());
//         }
//         fields_determined = DETERMINED_NOTHING;
//         expansion.symmetry = symmetry;
//     }
// 
//     /// Set transverse wavevector
//     void setKtran(dcomplex k)  {
//         if (expansion.initialized && (expansion.symmetric && k != 0.))
//             throw Exception("%1%: Cannot remove mode symmetry now -- invalidate the solver first", getId());
//         if (k != ktran) fields_determined = DETERMINED_NOTHING;
//         ktran = k;
//     }
// 
//     /// Return current mode polarization
//     ExpansionPW3D::Component getPolarization() const { return expansion.polarization; }
//     /// Set new mode polarization
//     void setPolarization(ExpansionPW3D::Component polarization) {
//         if (expansion.initialized) {
//             if (expansion.separated && polarization == ExpansionPW3D::E_UNSPECIFIED)
//                 throw Exception("%1%: Cannot remove polarizations separation now -- invalidate the solver first", getId());
//             if (!expansion.separated && polarization != ExpansionPW3D::E_UNSPECIFIED)
//                 throw Exception("%1%: Cannot add polarizations separation now -- invalidate the solver first", getId());
//         }
//         expansion.polarization = polarization;
//     }
// 
//     /**
//      * Get period
//      */
//     double getPeriod() {
//         bool not_initialized(!expansion.initialized);
//         if (not_initialized) expansion.init();
//         double result = (expansion.right - expansion.left) * (expansion.symmetric? 2. : 1.);
//         if (not_initialized) expansion.free();
//         return result;
//     }
// 
//     /**
//      * Get refractive index after expansion
//      */
//     DataVector<const Tensor3<dcomplex>> getRefractiveIndexProfile(const RectilinearMesh3D& dst_mesh,
//                                                                   InterpolationMethod interp=INTERPOLATION_DEFAULT);
// 
//   private:
// 
//     /**
//      * Get incident field vector for given polarization.
//      * \param polarization polarization of the perpendicularly incident light
//      * \param savidx pointer to which optionally save nonzero incident index
//      * \return incident field vector
//      */
//     cvector incidentVector(ExpansionPW3D::Component polarization, size_t* savidx=nullptr) {
//         size_t idx;
//         if (polarization == ExpansionPW3D::E_UNSPECIFIED)
//             throw BadInput(getId(), "Wrong incident polarization specified for the reflectivity computation");
//         if (expansion.symmetric) {
//             if (expansion.symmetry == ExpansionPW3D::E_UNSPECIFIED)
//                 expansion.symmetry = polarization;
//             else if (expansion.symmetry != polarization)
//                 throw BadInput(getId(), "Current symmetry is inconsistent with the specified incident polarization");
//         }
//         if (expansion.separated) {
//             expansion.polarization = polarization;
//             idx = expansion.iE(0);
//         } else {
//             idx = (polarization == ExpansionPW3D::E_TRAN)? expansion.iEx(0) : expansion.iEz(0);
//         }
//         if (savidx) *savidx = idx;
//         cvector incident(expansion.matrixSize(), 0.);
//         incident[idx] = 1.;
//         return incident;
//     }
// 
//     /**
//      * Compute sum of amplitudes for reflection/transmission coefficient
//      * \param amplitudes amplitudes to sum
//      */
//     double sumAmplitutes(const cvector& amplitudes) {
//         double result = 0.;
//         int N = getSize();
//         if (expansion.separated) {
//             if (expansion.symmetric) {
//                 for (int i = 0; i <= N; ++i)
//                     result += real(amplitudes[expansion.iE(i)]);
//                 result = 2.*result - real(amplitudes[expansion.iE(0)]);
//             } else {
//                 for (int i = -N; i <= N; ++i)
//                     result += real(amplitudes[expansion.iE(i)]);
//             }
//         } else {
//             if (expansion.symmetric) {
//                 for (int i = 0; i <= N; ++i)
//                     result += real(amplitudes[expansion.iEx(i)]) + real(amplitudes[expansion.iEz(i)]);
//                 result = 2.*result - real(amplitudes[expansion.iEx(0)]) - real(amplitudes[expansion.iEz(0)]);
//             } else {
//                 for (int i = -N; i <= N; ++i) {
//                     result += real(amplitudes[expansion.iEx(i)]) + real(amplitudes[expansion.iEz(i)]);
//                 }
//             }
//         }
//         return result;
//     }
// 
//   public:
// 
//     /**
//      * Get amplitudes of reflected diffraction orders
//      * \param polarization polarization of the perpendicularly incident light
//      * \param incidence incidence side
//      * \param savidx pointer to which optionally save nonzero incident index
//      */
//     cvector getReflectedAmplitudes(ExpansionPW3D::Component polarization, IncidentDirection incidence, size_t* savidx=nullptr);
// 
//     /**
//      * Get amplitudes of transmitted diffraction orders
//      * \param polarization polarization of the perpendicularly incident light
//      * \param incidence incidence side
//      * \param savidx pointer to which optionally save nonzero incident index
//      */
//     cvector getTransmittedAmplitudes(ExpansionPW3D::Component polarization, IncidentDirection incidence, size_t* savidx=nullptr);
// 
//     /**
//      * Get reflection coefficient
//      * \param polarization polarization of the perpendicularly incident light
//      * \param incidence incidence side
//      */
//     double getReflection(ExpansionPW3D::Component polarization, IncidentDirection incidence);
// 
//     /**
//      * Get reflection coefficient
//      * \param polarization polarization of the perpendicularly incident light
//      * \param incidence incidence side
//      */
//     double getTransmission(ExpansionPW3D::Component polarization, IncidentDirection incidence);
// 
//     /**
//      * Get electric field at the given mesh for reflected light.
//      * \param Ei incident field vector
//      * \param incident incidence direction
//      * \param dst_mesh target mesh
//      * \param method interpolation method
//      */
//     DataVector<Vec<3,dcomplex>> getReflectedFieldE(ExpansionPW3D::Component polarization, IncidentDirection incident,
//                                                    const MeshD<2>& dst_mesh, InterpolationMethod method) {
//         initCalculation();
//         return ReflectionSolver<Geometry3DCartesian>::getReflectedFieldE(incidentVector(polarization), incident, dst_mesh, method);
//     }
// 
//     /**
//      * Get magnetic field at the given mesh for reflected light.
//      * \param Ei incident field vector
//      * \param incident incidence direction
//      * \param dst_mesh target mesh
//      * \param method interpolation method
//      */
//     DataVector<Vec<3,dcomplex>> getReflectedFieldH(ExpansionPW3D::Component polarization, IncidentDirection incident,
//                                                    const MeshD<2>& dst_mesh, InterpolationMethod method) {
//         initCalculation();
//         return ReflectionSolver<Geometry3DCartesian>::getReflectedFieldH(incidentVector(polarization), incident, dst_mesh, method);
//     }
// 
//     /**
//      * Get light intensity for reflected light.
//      * \param Ei incident field vector
//      * \param incident incidence direction
//      * \param dst_mesh destination mesh
//      * \param method interpolation method
//      */
//     DataVector<double> getReflectedFieldIntensity(ExpansionPW3D::Component polarization, IncidentDirection incident,
//                                                   const MeshD<2>& dst_mesh, InterpolationMethod method) {
//         initCalculation();
//         return ReflectionSolver<Geometry3DCartesian>::getReflectedFieldIntensity(incidentVector(polarization), incident, dst_mesh, method);
//     }
// 
// 
//   protected:
// 
//     /// Insert mode to the list or return the index of the exiting one
//     size_t insertMode() {
//         Mode mode(this);
//         mode.k0 = k0; mode.klong = klong; mode.ktran = ktran;
//         mode.symmetry = expansion.symmetry; mode.polarization = expansion.polarization;
//         for (size_t i = 0; i != modes.size(); ++i)
//             if (modes[i] == mode) return i;
//         modes.push_back(mode);
//         outNeff.fireChanged();
//         outLightIntensity.fireChanged();
//         return modes.size()-1;
//     }
// 
//     size_t nummodes() const { return outNeff.size(); }
// 
//     /**
//      * Return mode effective index
//      * \param n mode number
//      */
//     dcomplex getEffectiveIndex(size_t n) {
//         if (n >= modes.size()) throw NoValue(EffectiveIndex::NAME);
//         return modes[n].klong / modes[n].k0;
//     }
// 
//     /**
//      * Compute electric field
//      * \param num mode number
//      * \param dst_mesh destination mesh
//      * \param method interpolation method
//      */
//     const DataVector<const Vec<3,dcomplex>> getE(size_t num, const MeshD<2>& dst_mesh, InterpolationMethod method);
// 
//     /**
//      * Compute magnetic field
//      * \param num mode number
//      * \param dst_mesh destination mesh
//      * \param method interpolation method
//      */
//     const DataVector<const Vec<3,dcomplex>> getH(size_t num, const MeshD<2>& dst_mesh, InterpolationMethod method);
// 
//     /**
//      * Compute light intensity
//      * \param num mode number
//      * \param dst_mesh destination mesh
//      * \param method interpolation method
//      */
//     const DataVector<const double> getIntensity(size_t num, const MeshD<2>& dst_mesh, InterpolationMethod method);
// 
//   public:
// 
//     /**
//      * Proxy class for accessing reflected fields
//      */
//     struct Reflected {
// 
//         /// Provider of the optical electric field
//         typename ProviderFor<OpticalElectricField,Geometry3DCartesian>::Delegate outElectricField;
// 
//         /// Provider of the optical magnetic field
//         typename ProviderFor<OpticalMagneticField,Geometry3DCartesian>::Delegate outMagneticField;
// 
//         /// Provider of the optical field intensity
//         typename ProviderFor<LightIntensity,Geometry3DCartesian>::Delegate outLightIntensity;
// 
//         /// Return one as the number of the modes
//         static size_t size() { return 1; }
// 
//         /**
//          * Construct proxy.
//          * \param wavelength incident light wavelength
//          * \param polarization polarization of the perpendicularly incident light
//          * \param side incidence side
//          */
//         Reflected(FourierReflection3D* parent, double wavelength, ExpansionPW3D::Component polarization, FourierReflection3D::IncidentDirection side):
//             outElectricField([=](size_t, const MeshD<2>& dst_mesh, InterpolationMethod method) -> DataVector<const Vec<3,dcomplex>> {
//                 parent->setWavelength(wavelength);
//                 return parent->getReflectedFieldE(polarization, side, dst_mesh, method); }, size),
//             outMagneticField([=](size_t, const MeshD<2>& dst_mesh, InterpolationMethod method) -> DataVector<const Vec<3,dcomplex>> {
//                 parent->setWavelength(wavelength);
//                 return parent->getReflectedFieldH(polarization, side, dst_mesh, method); }, size),
//             outLightIntensity([=](size_t, const MeshD<2>& dst_mesh, InterpolationMethod method) -> DataVector<const double> {
//                 parent->setWavelength(wavelength);
//                 return parent->getReflectedFieldIntensity(polarization, side, dst_mesh, method); }, size)
//         {}
//     };
// };
// 
// 
// }}} // namespace
// 
// #endif
// 
