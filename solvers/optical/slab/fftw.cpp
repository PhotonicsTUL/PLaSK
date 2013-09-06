#include "fft.h"

#ifdef USE_FFTW

#ifdef OPENMP_FOUND
#include <omp.h>
#endif

namespace plask { namespace solvers { namespace slab { namespace FFT {

// namespace detail {
//     struct FftwInitializer {
//         FftwInitializer() {
// #if defined(OPENMP_FOUND) && defined(USE_PARALLEL_FFT)
//             fftw_init_threads();
//             fftw_plan_with_nthreads(omp_get_max_threads());
// #endif
//         }
//         ~FftwInitializer() {
// #if defined(OPENMP_FOUND) && defined(USE_PARALLEL_FFT)
//             fftw_cleanup_threads();
// #else
//             fftw_cleanup();
// #endif
//         }
//     };
//     FftwInitializer fftwinitializer;
// }

Forward1D::Forward1D(): plan(nullptr) {}

Forward1D::Forward1D(Forward1D&& old):
    howmany(old.howmany), n(old.n),
    symmetry(old.symmetry),
    data(old.data), plan(old.plan) {
    old.plan = nullptr;
}

Forward1D& Forward1D::operator=(Forward1D&& old) {
    howmany = old.howmany; n = old.n;
    symmetry = old.symmetry;
    data = old.data; plan = old.plan;
    old.plan = nullptr;
    return *this;
}

Forward1D::Forward1D(size_t howmany, size_t n, Symmetry symmetry, dcomplex* data):
    howmany(howmany), n(n), symmetry(symmetry), data(data) {
    if (symmetry == SYMMETRY_NONE) {
        plan = fftw_plan_many_dft(1, &this->n, howmany,
                                  reinterpret_cast<fftw_complex*>(data), nullptr, howmany, 1,
                                  reinterpret_cast<fftw_complex*>(data), nullptr, howmany, 1,
                                  FFTW_FORWARD, FFTW_ESTIMATE);
    } else if (symmetry == SYMMETRY_EVEN) {
        static const fftw_r2r_kind kinds[] = { FFTW_REDFT10 };
        plan = fftw_plan_many_r2r(1, &this->n, 2*howmany,
                                  reinterpret_cast<double*>(data), nullptr, 2*howmany, 1,
                                  reinterpret_cast<double*>(data), nullptr, 2*howmany, 1,
                                  kinds, FFTW_ESTIMATE);
    } else
        throw NotImplemented("forward FFT for odd symmetry");
}

void Forward1D::execute() {
    if (!plan) throw CriticalException("No FFTW plan");
    fftw_execute(plan);
    double factor = (symmetry==SYMMETRY_NONE? 1. : 0.5) / n;
    for (size_t N = howmany*n, i = 0; i < N; ++i) data[i] *= factor;
}

void Forward1D::execute(dcomplex* data) {
    if (!plan) throw CriticalException("No FFTW plan");
    double factor;
    if (symmetry == SYMMETRY_NONE) {
        fftw_execute_dft(plan, reinterpret_cast<fftw_complex*>(data), reinterpret_cast<fftw_complex*>(data));
        factor = 1.0 / n;
    } else {
        fftw_execute_r2r(plan, reinterpret_cast<double*>(data), reinterpret_cast<double*>(data));
        factor = 0.5 / n;
    }
    for (size_t N = howmany*n, i = 0; i < N; ++i) data[i] *= factor;
}

Forward1D::~Forward1D() {
    if (plan) fftw_destroy_plan(plan);
}




Forward2D::Forward2D(): plan(nullptr) {}

Forward2D::Forward2D(Forward2D&& old):
    howmany(old.howmany), n1(old.n1), n2(old.n2),
    symmetry1(old.symmetry1), symmetry2(old.symmetry2),
    data(old.data), plan(old.plan) {
    old.plan = nullptr;
}

Forward2D& Forward2D::operator=(Forward2D&& old) {
    howmany = old.howmany; n1 = old.n1; n2 = old.n2;
    symmetry1 = old.symmetry1; symmetry2 = old.symmetry2;
    data = old.data; plan = old.plan;
    old.plan = nullptr;
    return *this;
}

Forward2D::Forward2D(size_t howmany, size_t n1, size_t n2, Symmetry symmetry1, Symmetry symmetry2, dcomplex* data):
    howmany(howmany), n1(n1), n2(n2), symmetry1(symmetry1), symmetry2(symmetry2), data(data) {
    if (symmetry1 == SYMMETRY_NONE && symmetry2 == SYMMETRY_NONE) {
        plan = fftw_plan_many_dft(2, &this->n1, howmany,
                                  reinterpret_cast<fftw_complex*>(data), nullptr, 1, n1*n2,
                                  reinterpret_cast<fftw_complex*>(data), nullptr, 1, n1*n2,
                                  FFTW_FORWARD, FFTW_ESTIMATE);
    } else if (symmetry1 == SYMMETRY_EVEN && symmetry2 == SYMMETRY_EVEN) {
        throw NotImplemented("FFTW even,even");//TODO
    } else if (symmetry1 == SYMMETRY_NONE && symmetry2 == SYMMETRY_EVEN) {
        throw NotImplemented("FFTW none,even");//TODO
    } else if (symmetry1 == SYMMETRY_EVEN && symmetry2 == SYMMETRY_NONE) {
        throw NotImplemented("FFTW even,none");//TODO
    } else
        throw NotImplemented("forward FFT for odd symmetry");
}

void Forward2D::execute() {
    if (!plan) throw CriticalException("No FFTW plan");
    fftw_execute(plan);
    double factor;
    if (symmetry1 == SYMMETRY_NONE && symmetry2 == SYMMETRY_NONE) factor = 0.5 / n1 / n2;
    for (size_t N = howmany*n1*n2, i = 0; i < N; ++i) data[i] *= factor;
}

void Forward2D::execute(dcomplex* data) {
    if (!plan) throw CriticalException("No FFTW plan");
    double factor;
    if (symmetry1 == SYMMETRY_NONE && symmetry2 == SYMMETRY_NONE) {
        fftw_execute_dft(plan, reinterpret_cast<fftw_complex*>(data), reinterpret_cast<fftw_complex*>(data));
        factor = 0.5 / n1 / n2;
    } else
        throw NotImplemented("Forward2D");
    for (size_t N = howmany*n1*n2, i = 0; i < N; ++i) data[i] *= factor;
}

Forward2D::~Forward2D() {
    if (plan) fftw_destroy_plan(plan);
}




Backward1D::Backward1D(): plan(nullptr) {}

Backward1D::Backward1D(Backward1D&& old):
    howmany(old.howmany), n(old.n),
    symmetry(old.symmetry),
    data(old.data), plan(old.plan) {
    old.plan = nullptr;
}

Backward1D& Backward1D::operator=(Backward1D&& old) {
    howmany = old.howmany; n = old.n;
    symmetry = old.symmetry;
    data = old.data; plan = old.plan;
    old.plan = nullptr;
    return *this;
}

Backward1D::Backward1D(size_t howmany, size_t n, Symmetry symmetry, dcomplex* data):
    howmany(howmany), n(n), symmetry(symmetry), data(data) {
    if (symmetry == SYMMETRY_NONE) {
        plan = fftw_plan_many_dft(1, &this->n, howmany,
                                  reinterpret_cast<fftw_complex*>(data), nullptr, howmany, 1,
                                  reinterpret_cast<fftw_complex*>(data), nullptr, howmany, 1,
                                  FFTW_BACKWARD, FFTW_ESTIMATE);
    } else if (symmetry == SYMMETRY_EVEN) {
        static const fftw_r2r_kind kinds[] = { FFTW_REDFT01 };
        plan = fftw_plan_many_r2r(1, &this->n, 2*howmany,
                                  reinterpret_cast<double*>(data), nullptr, 2*howmany, 1,
                                  reinterpret_cast<double*>(data), nullptr, 2*howmany, 1,
                                  kinds, FFTW_ESTIMATE);
    } else {
        static const fftw_r2r_kind kinds[] = { FFTW_RODFT01 };
        plan = fftw_plan_many_r2r(1, &this->n, 2*howmany,
                                  reinterpret_cast<double*>(data), nullptr, 2*howmany, 1,
                                  reinterpret_cast<double*>(data), nullptr, 2*howmany, 1,
                                  kinds, FFTW_ESTIMATE);
    }
}

void Backward1D::execute() {
    if (!plan) throw CriticalException("No FFTW plan");
    fftw_execute(plan);
}
void Backward1D::execute(dcomplex* data) {
    if (!plan) throw CriticalException("No FFTW plan");
    if (symmetry == SYMMETRY_NONE)
        fftw_execute_dft(plan, reinterpret_cast<fftw_complex*>(data), reinterpret_cast<fftw_complex*>(data));
    else
        fftw_execute_r2r(plan, reinterpret_cast<double*>(data), reinterpret_cast<double*>(data));
}

Backward1D::~Backward1D() {
    if (plan) fftw_destroy_plan(plan);
}




Backward2D::Backward2D(): plan(nullptr) {}

Backward2D::Backward2D(Backward2D&& old):
    howmany(old.howmany), n1(old.n1), n2(old.n2),
    symmetry1(old.symmetry1), symmetry2(old.symmetry2),
    data(old.data), plan(old.plan) {
    old.plan = nullptr;
}

Backward2D& Backward2D::operator=(Backward2D&& old) {
    howmany = old.howmany; n1 = old.n1; n2 = old.n2;
    symmetry1 = old.symmetry1; symmetry2 = old.symmetry2;
    data = old.data; plan = old.plan;
    old.plan = nullptr;
    return *this;
}

Backward2D::Backward2D(size_t howmany, size_t n1, size_t n2, Symmetry symmetry1, Symmetry symmetry2, dcomplex* data):
    howmany(howmany), n1(n1), n2(n2), symmetry1(symmetry1), symmetry2(symmetry2), data(data) {
    plan = nullptr; //TODO
}

void Backward2D::execute() {
    if (!plan) throw CriticalException("No FFTW plan");
    fftw_execute(plan);
}

void Backward2D::execute(dcomplex* data) {
    if (!plan) throw CriticalException("No FFTW plan");
//     fftw_execute(plan);
}

Backward2D::~Backward2D() {
    if (plan) fftw_destroy_plan(plan);
}


}}}} // namespace plask::solvers::slab

#endif // USE_FFTW
