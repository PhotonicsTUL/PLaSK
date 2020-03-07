#ifndef PLASK__SOLVER_SLAB_XANCE_H
#define PLASK__SOLVER_SLAB_XANCE_H

#include "matrices.h"
#include "transfer.h"
#include "solver.h"


namespace plask { namespace optical { namespace slab {

/**
 * Base class for all solvers using reflection matrix method.
 */
struct PLASK_SOLVER_API XanceTransfer: public Transfer {

    /// The structure holding the set of diagonalized fields at the layer boundaries
    struct FieldsDiagonalized {
        cvector E0, H0, Ed, Hd;
    };

  protected:

    cmatrix Y;                                  ///< Admittance / impedance matrix

    bool needAllY;                              ///< Do we need to keep all Y matrices?

    std::vector<FieldsDiagonalized> fields;     ///< Vector of fields computed for each layer

    std::vector<cmatrix> memY;                  ///< admittance matrices for each layer

    double get_d(size_t n, double& z) {       ///< get layer thickness and adjust z
        double d = (n == 0 || std::size_t(n) == solver->vbounds->size())?
            solver->vpml.dist :
            solver->vbounds->at(n) - solver->vbounds->at(n-1);
        if (std::ptrdiff_t(n) >= solver->interface) z = d - z;
        else if (n == 0) z += d;
        return d;
    }

  public:

    cvector getTransmissionVector(const cvector& incident, IncidentDirection side) override;

    XanceTransfer(SlabBase* solver, Expansion& expansion);

  protected:

    cvector getFieldVectorE(double z, std::size_t n) override;

    cvector getFieldVectorH(double z, std::size_t n) override;

    /**
     * Store the Y matrix for the layer prepared before
     * \param n layer number
     */
    void storeY(size_t n);

    /**
     * Get the Y matrix for n-th layer
     * \param n layer number
     */
    const cmatrix& getY(std::size_t n) {
        if (memY.size() == solver->stack.size() && needAllY)
            return memY[n];
        else
            throw CriticalException("{0}: Y matrices are not stored", solver->getId());
    }

    /// Determine the y1 efficiently
    inline void get_y1(const cdiagonal& gamma, double d, cdiagonal& y1) const {
        const std::size_t N = gamma.size();
        assert(y1.size() == N);
        for (std::size_t i = 0; i < N; i++) {
            dcomplex t = tanh(I*gamma[i]*d);
            if (isinf(real(t)) || isinf(imag(t))) y1[i] = 0.;
            else if (abs(t) < SMALL)
                throw ComputationError(solver->getId(),
                                       "Matrix y1 has some infinite value (try changing wavelength or layer thickness a bit)");
            else y1[i] = 1. / t;
        }
    }

    /// Determine the y2 efficiently
    inline void get_y2(const cdiagonal& gamma, double d, cdiagonal& y2) const {
        const std::size_t N = gamma.size();
        assert(y2.size() == N);
        for (std::size_t i = 0; i < N; i++) {
            dcomplex s = sinh(I*gamma[i]*d);
            if (isinf(real(s)) || isinf(imag(s))) y2[i] = 0.;
            else if (abs(s) < SMALL)
                throw ComputationError(solver->getId(),
                                       "Matrix y2 has some infinite value (try changing wavelength or layer thickness a bit)");
            else y2[i] = - 1. / s;
        }
    }

    double integrateField(WhichField field, size_t n, double z1, double z2) override;
};


}}} // namespace plask::optical::slab

#endif // PLASK__SOLVER_SLAB_XANCE_H
