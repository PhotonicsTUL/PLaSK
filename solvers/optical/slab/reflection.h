#ifndef PLASK__SOLVER_SLAB_REFLECTION_H
#define PLASK__SOLVER_SLAB_REFLECTION_H

#include "matrices.h"
#include "transfer.h"
#include "solver.h"


namespace plask { namespace optical { namespace slab {

/**
 * Base class for all solvers using reflection matrix method.
 */
struct PLASK_SOLVER_API ReflectionTransfer: public Transfer {

    /// Struct containing data for computing field in a layer
    struct LayerFields {
        cvector F, B;
    };

  protected:

    cmatrix P;                                  ///< Current reflection matrix
    enum {
        STORE_NONE,
        STORE_LAST,
        STORE_ALL
    } storeP;                                   ///< Do we need to keep the P matrices for both sides?

    std::vector<LayerFields> fields;            ///< Vector of fields computed for each layer

  private:

    cdiagonal phas;                             ///< Current phase dist matrix
    int* ipiv;                                  ///< Pivot vector
    std::vector<cmatrix> memP;                  ///< Reflection matrices from each side

    void saveP(size_t n) {
        if (memP[n].rows() == P.rows() && memP[n].cols() == P.cols())
            memcpy(memP[n].data(), P.data(), P.rows() * P.cols() * sizeof(dcomplex));
        else
            memP[n] = P.copy();
    }

    void adjust_z(size_t n, double& z) {
        if (std::ptrdiff_t(n) >= solver->interface) {
            z = - z;
            if (n != 0 && n != solver->vbounds->size())
                z += solver->vbounds->at(n) - solver->vbounds->at(n-1);
        }
    }

    void adjust_z(size_t n, double& z1, double& z2) {
        if (std::ptrdiff_t(n) >= solver->interface) {
            double zl = z1;
            z1 = - z2; z2 = - zl;
            if (n != 0 && n != solver->vbounds->size()) {
                double d = solver->vbounds->at(n) - solver->vbounds->at(n-1);
                z1 += d; z2 += d;
            }
        }
    }

  public:

    ReflectionTransfer(SlabBase* solver, Expansion& expansion);

    ~ReflectionTransfer();

    cvector getReflectionVector(const cvector& incident, IncidentDirection direction) override;

    cvector getTransmissionVector(const cvector& incident, IncidentDirection side) override;

  protected:

    void getFinalMatrix() override;

    void determineFields() override;

    void determineReflectedFields(const cvector& incident, IncidentDirection side) override;

    cvector getFieldVectorE(double z, std::size_t n) override;

    cvector getFieldVectorH(double z, std::size_t n) override;

    /**
     * Get admittance (A) and discontinuity (M) matrices for half of the structure
     * \param start start of the transfer
     * \param end end of the transfer
     * \param add if \c true then M matrix is added to the previous value
     * \param mfac factor to multiply M matrix befere addition
     */
    void getAM(size_t start, size_t end, bool add, double mfac=1.);

    /**
     * Find reflection matrix for the part of the structure
     * \param start starting layer
     * \param end last layer (reflection matrix is computed for this layer)
     * \param emitting should the reflection matrix in the first layer be 0?
     * \param store where the final P matrix should be stored if so?
     */
    void findReflection(std::size_t start, std::size_t end, bool emitting, int store=0);

    double integrateField(WhichField field, size_t n, double z1, double z2) override;
};


}}} // namespace plask::optical::slab

#endif // PLASK__SOLVER_SLAB_REFLECTION_H
