#ifndef PLASK__SOLVER_SLAB_EXPANSION_PW2D_H
#define PLASK__SOLVER_SLAB_EXPANSION_PW2D_H

#include <plask/plask.hpp>

#include "../expansion.hpp"
#include "../meshadapter.hpp"
#include "fft.hpp"

namespace plask { namespace optical { namespace slab {

struct FourierSolver2D;

struct PLASK_SOLVER_API ExpansionPW2D: public Expansion {

    dcomplex beta,                      ///< Longitudinal wavevector [1/µm]
             ktran;                     ///< Transverse wavevector [1/µm]

    size_t N;                           ///< Number of expansion coefficients
    size_t nN;                          ///< Number of of required coefficients for material parameters
    double left;                        ///< Left side of the sampled area
    double right;                       ///< Right side of the sampled area
    bool periodic;                      ///< Indicates if the geometry is periodic (otherwise use PMLs)
    bool initialized;                   ///< Expansion is initialized

    Component symmetry;                 ///< Indicates symmetry if `symmetric`
    Component polarization;             ///< Indicates polarization if `separated`

    size_t pil,                         ///< Index of the beginning of the left PML
           pir;                         ///< Index of the beginning of the right PML

    struct Coeffs {
        DataVector<dcomplex> zz, rxx, yy, ryy, zx, rzx;
    };
    /// Cached permittivity expansion coefficients
    std::vector<Coeffs> coeffs;

    struct CoeffMatrices {
        cmatrix exx, reyy, ezx;
    };
    /// Cached permittivity expansion coefficient matrixess
    std::vector<CoeffMatrices> coeff_matrices;

    /// Cached permeability expansion coefficient matrices
    cmatrix coeff_matrix_mxx, coeff_matrix_rmyy;

    /// Information if the layer is diagonal
    std::vector<bool> diagonals;

    /// Mesh for getting material data
    shared_ptr<RectangularMesh<2>> mesh;

    /// Boundaries mesh
    shared_ptr<MeshAxis> original_mesh;

    /**
     * Create new expansion
     * \param solver solver which performs calculations
     */
    ExpansionPW2D(FourierSolver2D* solver);

    /// Indicates if the expansion is a symmetric one
    bool symmetric() const { return symmetry != E_UNSPECIFIED; }

    /// Indicates whether TE and TM modes can be separated
    bool separated() const { return polarization != E_UNSPECIFIED; }

    /**
     * Init expansion
     * \param compute_coeffs compute material coefficients
     */
    void init();

    /// Free allocated memory
    void reset();

    bool diagonalQE(size_t l) const override {
        return diagonals[l];
    }

    size_t matrixSize() const override { return separated()? N : 2*N; }

    void getMatrices(size_t l, cmatrix& RE, cmatrix& RH) override;

    void prepareField() override;

    void cleanupField() override;

    LazyData<Vec<3,dcomplex>> getField(size_t l,
                                       const shared_ptr<const typename LevelsAdapter::Level>& level,
                                       const cvector& E, const cvector& H) override;

    LazyData<Tensor3<dcomplex>> getMaterialNR(size_t l,
                                              const shared_ptr<const typename LevelsAdapter::Level>& level,
                                              InterpolationMethod interp=INTERPOLATION_DEFAULT) override;

    double integrateField(WhichField field, size_t l, const cvector& E, const cvector& H) override;

    double integratePoyntingVert(const cvector& E, const cvector& H) override;

    void getDiagonalEigenvectors(cmatrix& Te, cmatrix Te1, const cmatrix& RE, const cdiagonal& gamma) override;

  private:

    DataVector<Vec<3,dcomplex>> field;
    FFT::Backward1D fft_x, fft_yz;

    void add_coeffs(int start, int end, double b, double l, double r, DataVector<dcomplex>& dst, dcomplex val) {
            for (int k = start; k != end; ++k) {
                size_t j = (k>=0)? k : k + nN;
                dcomplex ff = (j)? (dcomplex(0., 0.5/PI/k) * (exp(dcomplex(0., -b*k*r)) - exp(dcomplex(0., -b*k*l)))) : ((r-l) * b*(0.5/PI));
                dst[j] += val * ff;
            }
    }

    void make_permeability_matrices(cmatrix& work);

  protected:

    DataVector<dcomplex> mag;               ///< Magnetic permeability coefficients (used with for PMLs)
    DataVector<dcomplex> rmag;              ///< Inverted magnetic permeability coefficients (used with for PMLs)

    FFT::Forward1D matFFT;                  ///< FFT object for material coefficients

    void beforeLayersIntegrals(double lam, double glam) override;

    void layerIntegrals(size_t layer, double lam, double glam) override;

    Tensor3<dcomplex> getEpsilon(const shared_ptr<GeometryD<2>>& geometry, size_t layer, double maty,
                                 double lam, double glam, size_t j) {
        double T = 0., W = 0., C = 0.;
        for (size_t k = 0, v = j * solver->verts->size(); k != mesh->vert()->size(); ++v, ++k) {
            if (solver->stack[k] == layer) {
                double w = (k == 0 || k == mesh->vert()->size()-1)? 1e-6 : solver->vbounds->at(k) - solver->vbounds->at(k-1);
                T += w * temperature[v]; C += w * carriers[v]; W += w;
            }
        }
        T /= W;
        C /= W;
        Tensor3<dcomplex> nr;
        {
            OmpLockGuard<OmpNestLock> lock; // this must be declared before `material` to guard its destruction
            auto material = geometry->getMaterial(vec(mesh->tran()->at(j),maty));
            lock = material->lock();
            nr = material->NR(lam, T, C);
            if (isnan(nr.c00) || isnan(nr.c11) || isnan(nr.c22) || isnan(nr.c01))
                throw BadInput(solver->getId(), "Complex refractive index (NR) for {} is NaN at lam={}nm, T={}K, n={}/cm3",
                               material->name(), lam, T, C);
        }
        if (nr.c01 != 0.) {
            if (symmetric()) throw BadInput(solver->getId(), "Symmetry not allowed for structure with non-diagonal NR tensor");
            if (separated()) throw BadInput(solver->getId(), "Single polarization not allowed for structure with non-diagonal NR tensor");
        }
        if (gain_connected && solver->lgained[layer]) {
            auto roles = geometry->getRolesAt(vec(mesh->tran()->at(j),maty));
            if (roles.find("QW") != roles.end() || roles.find("QD") != roles.end() || roles.find("gain") != roles.end()) {
                Tensor2<double> g = 0.; W = 0.;
                for (size_t k = 0, v = j * solver->verts->size(); k != mesh->vert()->size(); ++v, ++k) {
                    if (solver->stack[k] == layer) {
                        double w = (k == 0 || k == mesh->vert()->size()-1)? 1e-6 : solver->vbounds->at(k) - solver->vbounds->at(k-1);
                        g += w * gain[v]; W += w;
                    }
                }
                Tensor2<double> ni = glam * g/W * (0.25e-7/PI);
                nr.c00.imag(ni.c00); nr.c11.imag(ni.c00); nr.c22.imag(ni.c11); nr.c01.imag(0.);
            }
        }
        nr.sqr_inplace();
        return nr;
    }

  public:

    dcomplex getBeta() const { return beta; }
    void setBeta(dcomplex b) {
        if (b != beta) {
            beta = b;
            solver->clearFields();
        }
    }

    dcomplex getKtran() const { return ktran; }
    void setKtran(dcomplex k) {
        if (k != ktran) {
            ktran = k;
            solver->clearFields();
        }
    }

    Component getSymmetry() const { return symmetry; }
    void setSymmetry(Component sym) {
        if (sym != symmetry) {
            symmetry = sym;
            solver->clearFields();
            solver->recompute_integrals = true;
        }
    }

    Component getPolarization() const { return polarization; }
    void setPolarization(Component pol);

    const DataVector<dcomplex>& epszz(size_t l) { return coeffs[l].zz; }            ///< Get \f$ \varepsilon_{zz} \f$
    const DataVector<dcomplex>& epsyy(size_t l) { return coeffs[l].yy; }            ///< Get \f$ \varepsilon_{yy} \f$
    const DataVector<dcomplex>& repsxx(size_t l) { return coeffs[l].rxx; }          ///< Get \f$ \varepsilon_{xx}^{-1} \f$
    const DataVector<dcomplex>& repsyy(size_t l) { return coeffs[l].ryy; }          ///< Get \f$ \varepsilon_{yy}^{-1} \f$
    const DataVector<dcomplex>& epszx(size_t l) { return coeffs[l].zx; }            ///< Get \f$ \varepsilon_{zx} \f$
    const DataVector<dcomplex>& repszx(size_t l) { return coeffs[l].rzx; }          ///< Get \f$ \varepsilon_{zx}^{-1} \f$
    const DataVector<dcomplex>& muzz() { return mag; }                              ///< Get \f$ \mu_{zz} \f$
    const DataVector<dcomplex>& muxx() { return mag; }                              ///< Get \f$ \mu_{xx} \f$
    const DataVector<dcomplex>& muyy() { return mag; }                              ///< Get \f$ \mu_{yy} \f$
    const DataVector<dcomplex>& rmuzz() { return rmag; }                            ///< Get \f$ \mu_{zz}^{-1} \f$
    const DataVector<dcomplex>& rmuxx() { return rmag; }                            ///< Get \f$ \mu_{xx}^{-1} \f$
    const DataVector<dcomplex>& rmuyy() { return rmag; }                            ///< Get \f$ \mu_{yy}^{-1} \f$

    dcomplex epszz(size_t l, int i) { return coeffs[l].zz[(i>=0)?i:i+nN]; }         ///< Get element of \f$ \varepsilon_{zz} \f$
    dcomplex epsyy(size_t l, int i) { return coeffs[l].yy[(i>=0)?i:i+nN]; }         ///< Get element of \f$ \varepsilon_{yy} \f$
    dcomplex repsxx(size_t l, int i) { return coeffs[l].rxx[(i>=0)?i:i+nN]; }       ///< Get element of \f$ \varepsilon_{xx}^{-1} \f$
    dcomplex repsyy(size_t l, int i) { return coeffs[l].ryy[(i>=0)?i:i+nN]; }       ///< Get element of \f$ \varepsilon_{yy}^{-1} \f$
    dcomplex epszx(size_t l, int i) { return coeffs[l].zx[(i>=0)?i:i+nN]; }         ///< Get element of \f$ \varepsilon_{zx} \f$
    dcomplex epsxz(size_t l, int i) { return conj(coeffs[l].zx[(i>=0)?i:i+nN]); }   ///< Get element of \f$ \varepsilon_{zx} \f$
    dcomplex repszx(size_t l, int i) { return coeffs[l].rzx[(i>=0)?i:i+nN]; }       ///< Get element of \f$ \varepsilon_{zx}^{-1} \f$
    dcomplex iepsxz(size_t l, int i) { return conj(coeffs[l].rzx[(i>=0)?i:i+nN]); } ///< Get element of \f$ \varepsilon_{zx}^{-1} \f$
    dcomplex muzz(int i) { return mag[(i>=0)?i:i+nN]; }                             ///< Get element of \f$ \mu_{zz} \f$
    dcomplex muxx(int i) { return mag[(i>=0)?i:i+nN]; }                             ///< Get element of \f$ \mu_{xx} \f$
    dcomplex muyy(int i) { return mag[(i>=0)?i:i+nN]; }                             ///< Get element of \f$ \mu_{yy} \f$
    dcomplex rmuzz(int i) { return rmag[(i>=0)?i:i+nN]; }                           ///< Get element of \f$ \mu_{zz}^{-1} \f$
    dcomplex rmuxx(int i) { return rmag[(i>=0)?i:i+nN]; }                           ///< Get element of \f$ \mu_{xx}^{-1} \f$
    dcomplex rmuyy(int i) { return rmag[(i>=0)?i:i+nN]; }                           ///< Get element of \f$ \mu_{yy}^{-1} \f$

    size_t iEx(int i) { return 2 * ((i>=0)?i:i+N); }        ///< Get \f$ E_x \f$ index
    size_t iEz(int i) { return 2 * ((i>=0)?i:i+N) + 1; }    ///< Get \f$ E_z \f$ index
    size_t iHx(int i) { return 2 * ((i>=0)?i:i+N) + 1; }    ///< Get \f$ H_x \f$ index
    size_t iHz(int i) { return 2 * ((i>=0)?i:i+N); }        ///< Get \f$ H_z \f$ index
    size_t iEH(int i) { return (i>=0)?i:i+N; }              ///< Get \f$ E \f$ or \f$ H \f$ index for separated equations
};

}}} // namespace plask::optical::slab

#endif // PLASK__SOLVER_SLAB_EXPANSION_PW2D_H
