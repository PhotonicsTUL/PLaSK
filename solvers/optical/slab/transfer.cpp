#include "transfer.h"
#include "diagonalizer.h"
#include "expansion.h"
#include "solver.h"
#include "fortran.h"

namespace plask { namespace optical { namespace slab {

Transfer::Transfer(SlabBase* solver, Expansion& expansion):
    solver(solver),
    diagonalizer(new SimpleDiagonalizer(&expansion)),   //TODO add other diagonalizer types
    fields_determined(DETERMINED_NOTHING)
{
    // Reserve space for matrix multiplications...
    const std::size_t N0 = diagonalizer->source()->matrixSize();
    const std::size_t N = diagonalizer->matrixSize();
    M = cmatrix(N0, N0);
    temp = cmatrix(N,N);

    // ...and eigenvalues determination
    evals = aligned_new_array<dcomplex>(N0);
    rwrk = aligned_new_array<double>(2*N0);
    lwrk = max(std::size_t(2*N0), N0*N0);
    wrk = aligned_new_array<dcomplex>(lwrk);

    // Nothing found so far
    fields_determined = DETERMINED_NOTHING;
    interface_field = nullptr;
}

Transfer::~Transfer()
{
    // no need for aligned_delete_array becase array memebers have trivial destructor
    aligned_free<dcomplex>(evals); evals = nullptr;
    aligned_free<double>(rwrk); rwrk = nullptr;
    aligned_free<dcomplex>(wrk); wrk = nullptr;
}


void Transfer::initDiagonalization() {
    // Get new coefficients if needed
    solver->computeIntegrals();
    this->diagonalizer->initDiagonalization();
}


dcomplex Transfer::determinant()
{
    // We change the matrices M and A so we will have to find the new fields
    fields_determined = DETERMINED_NOTHING;

    initDiagonalization();

    // Obtain admittance
    getFinalMatrix();

    const std::size_t N = M.rows();

    // This is probably expensive but necessary check to avoid hangs
    const std::size_t NN = N*N;
    for (std::size_t i = 0; i < NN; i++) {
        if (isnan(real(M[i])) || isnan(imag(M[i])))
            throw ComputationError(solver->getId(), "NaN in discontinuity matrix");
    }

    // Find the eigenvalues of M using LAPACK
    int info;
    zgeev('N', 'N', int(N), M.data(), int(N), evals, nullptr, 1, nullptr, 1, wrk, int(lwrk), rwrk, info);
    if (info != 0) throw ComputationError(solver->getId(), "eigenvalue determination failed");

    //TODO add some consideration for degenerate modes
    // Find the smallest eigenvalue
    dcomplex result;
    double min_mag = 1e32;
    for (std::size_t i = 0; i < N; i++) {
        dcomplex val = evals[i];
        double mag = abs2(val);
        if (mag < min_mag) { min_mag = mag; result = val; }
    }
    // // Find the determinant
    // dcomplex result = 1.;
    // for (int i = 0; i < N; i++) {
    //     result *= evals[i];
    // }
    // result = log(result);

    interface_field = nullptr;

    return result;
}


const_cvector Transfer::getInterfaceVector()
{
    const std::size_t N = M.rows();

    // Check if the necessary memory is already allocated
    if (interface_field_matrix.rows() != N) {
        interface_field_matrix = cmatrix(N,N);
        interface_field = nullptr;
    }

    // If the field already found, don't compute again
    if (!interface_field) {

        // We change the matrices M and A so we will have to find the new fields
        fields_determined = DETERMINED_NOTHING;

        initDiagonalization();

        // Obtain admittance
        getFinalMatrix();

        // Find the eigenvalues of M using LAPACK
        int info;
        zgeev('N', 'V', int(N), M.data(), int(N), evals, nullptr, 1, interface_field_matrix.data(), int(N), wrk, int(lwrk), rwrk, info);
        if (info != 0) throw ComputationError(solver->getId(), "Interface field: zgeev failed");

        // Find the number of the smallest eigenvalue
        double min_mag = 1e32;
        std::size_t n;
        for (std::size_t i = 0; i < N; i++) {
            double mag = abs2(evals[i]);
            if (mag < min_mag) { min_mag = mag; n = i; }
        }

        // Error handling
        if (min_mag > solver->root.tolf_max * solver->root.tolf_max)
            throw BadInput(solver->getId(), "Interface field: determinant not sufficiently close to 0 (det={})", str(evals[n]));

        // Chose the eigenvector corresponding to the smallest eigenvalue
        interface_field = interface_field_matrix.data() + n*N;
    }

    return DataVector<const dcomplex>(interface_field, N);
}


LazyData<Vec<3,dcomplex>> Transfer::computeFieldE(double power, const shared_ptr<const Mesh> &dst_mesh, InterpolationMethod method, bool reflected)
{
    double fact = sqrt(2e-3 * power);
    double zlim = solver->vpml.dist + solver->vpml.size;
    DataVector<Vec<3,dcomplex>> destination(dst_mesh->size());
    auto levels = makeLevelsAdapter(dst_mesh);
    diagonalizer->source()->initField(Expansion::FIELD_E, method);
    while (auto level = levels->yield()) {
        double z = level->vpos();
        const std::size_t n = solver->getLayerFor(z);
        if (!reflected) {
            if (n == 0 && z < -zlim) z = -zlim;
            else if (n == solver->stack.size()-1 && z > zlim) z = zlim;
        }
        cvector E = getFieldVectorE(z, n);
        cvector H = getFieldVectorH(z, n);
        if (std::ptrdiff_t(n) >= solver->interface) for (auto& h: H) h = -h;
        size_t layer = solver->stack[n];
        auto dest = fact * diagonalizer->source()->getField(layer, level, E, H);
        for (size_t i = 0; i != level->size(); ++i) destination[level->index(i)] = dest[i];
    }
    diagonalizer->source()->cleanupField();
    return destination;
}

LazyData<Vec<3,dcomplex>> Transfer::computeFieldH(double power, const shared_ptr<const Mesh>& dst_mesh, InterpolationMethod method, bool reflected)
{
    double fact = 1./Z0 * sqrt(2e-3 * power);
    double zlim = solver->vpml.dist + solver->vpml.size;
    DataVector<Vec<3,dcomplex>> destination(dst_mesh->size());
    auto levels = makeLevelsAdapter(dst_mesh);
    diagonalizer->source()->initField(Expansion::FIELD_H, method);
    while (auto level = levels->yield()) {
        double z = level->vpos();
        size_t n = solver->getLayerFor(z);
        if (!reflected) {
            if (n == 0 && z < -zlim) z = -zlim;
            else if (n == solver->stack.size()-1 && z > zlim) z = zlim;
        }
        cvector E = getFieldVectorE(z, n);
        cvector H = getFieldVectorH(z, n);
        if (std::ptrdiff_t(n) >= solver->interface) for (auto& h: H) h = -h;
        size_t layer = solver->stack[n];
        auto dest = fact * diagonalizer->source()->getField(layer, level, E, H);
        for (size_t i = 0; i != level->size(); ++i) destination[level->index(i)] = dest[i];
    }
    diagonalizer->source()->cleanupField();
    return destination;
}


cvector Transfer::getFieldVectorE(double z) {
    determineFields();
    const std::size_t n = solver->getLayerFor(z);
    return getFieldVectorE(z, n);
}

cvector Transfer::getFieldVectorH(double z) {
    determineFields();
    const std::size_t n = solver->getLayerFor(z);
    cvector H = getFieldVectorH(z, n);
    if (std::ptrdiff_t(n) >= solver->interface) for (auto& h: H) h = -h;
    return H;
}

cvector Transfer::getScatteredFieldVectorE(const cvector& incident, IncidentDirection side, double z) {
    determineReflectedFields(incident, side);
    return getFieldVectorE(z, solver->getLayerFor(z));
}

cvector Transfer::getScatteredFieldVectorH(const cvector& incident, IncidentDirection side, double z) {
    determineReflectedFields(incident, side);
    return getFieldVectorH(z, solver->getLayerFor(z));
}



double Transfer::getIntegralEE(double z1, double z2) {
    determineFields();
    if (z1 > z2) std::swap(z1, z2);
    size_t end = solver->getLayerFor(z2);
    if (is_zero(z2)) --end;
    double result = 0.;
    for (size_t n = solver->getLayerFor(z1); n <= end; ++n) {
        result += integrateEE(n, z1, (n != end)? solver->vbounds->at(n+1) - solver->vbounds->at(n) : z2);
        z1 = 0.;
    }
    return 0.5 * result;
}

double Transfer::getIntegralHH(double z1, double z2) {
    determineFields();
    if (z1 > z2) std::swap(z1, z2);
    size_t end = solver->getLayerFor(z2);
    if (is_zero(z2)) --end;
    double result = 0.;
    for (size_t n = solver->getLayerFor(z1); n <= end; ++n) {
        result += integrateHH(n, z1, (n != end)? solver->vbounds->at(n+1) - solver->vbounds->at(n) : z2);
        z1 = 0.;
    }
    return 0.5 * result;
}


}}} // namespace plask::optical::slab
