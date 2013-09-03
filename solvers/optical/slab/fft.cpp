#include "fft.h"
#include <plask/optical/slab/config.h>

#ifdef USE_FFTW

#ifdef OPENMP_FOUND
#include <omp.h>
#endif

#include <fftw3.h>

namespace plask { namespace solvers { namespace slab {

FFT::FFT() {
#if defined(OPENMP_FOUND) && defined(USE_PARALLEL_FFT)
    static bool fresh = true;
    if (fresh) {
        fftw_init_threads();
        fresh = false;
    }
    fftw_plan_with_nthreads(omp_get_max_threads());
#endif
};


FFT::~FFT() {
#if defined(OPENMP_FOUND) && defined(USE_PARALLEL_FFT)
    fftw_cleanup_threads();
#else
    fftw_cleanup();
#endif
}

void FFT::forward(size_t howmany, size_t n, dcomplex* data, Symmetry symmetry)
{
    if (symmetry == SYMMETRY_NONE) {

        int ranks[] = { n };
        fftw_plan plan = fftw_plan_many_dft(1, ranks, howmany,
                                        reinterpret_cast<fftw_complex*>(data), NULL, howmany, 1,
                                        reinterpret_cast<fftw_complex*>(data), NULL, howmany, 1,
                                        FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);

        double factor = 1. / n;
        for (size_t N = howmany*n, i = 0; i < N; i++) data[i] *= factor;

    } else if (symmetry == SYMMETRY_EVEN) {

        int ranks[] = { n };
        fftw_r2r_kind kinds[] = { FFTW_REDFT10 };
        fftw_plan plan = fftw_plan_many_r2r(1, ranks, 2*howmany,
                                            reinterpret_cast<double*>(data), NULL, 2*howmany, 1,
                                            reinterpret_cast<double*>(data), NULL, 2*howmany, 1,
                                            kinds, FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);

        double factor = 0.5 / n;
        for (size_t N = howmany*n, i = 0; i < N; i++) data[i] *= factor;

    } else
        throw NotImplemented("forward FFT for odd symmetry");
}


void FFT::forward(size_t howmany, size_t n1, size_t n2, dcomplex* data, Symmetry symmetry1, Symmetry symmetry2)
{
    if (symmetry1 == SYMMETRY_NONE && symmetry2 == SYMMETRY_NONE) {

        int ranks[] = { n1, n2 };
        fftw_plan plan = fftw_plan_many_dft(1, ranks, howmany,
                                            reinterpret_cast<fftw_complex*>(data), NULL, 1, n1*n2,
                                            reinterpret_cast<fftw_complex*>(data), NULL, 1, n1*n2,
                                            FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);
        double factor = 1. / n1 / n2;
        for (size_t N = howmany*n1*n2, i = 0; i < N; i++) data[i] *= factor;

    } else if (symmetry1 == SYMMETRY_EVEN && symmetry2 == SYMMETRY_EVEN) {
        //TODO
    } else if (symmetry1 == SYMMETRY_NONE && symmetry2 == SYMMETRY_EVEN) {
        //TODO
    } else if (symmetry1 == SYMMETRY_EVEN && symmetry2 == SYMMETRY_NONE) {
        //TODO
    } else
        throw NotImplemented("forward FFT for odd symmetry");
}


void FFT::backward(size_t howmany, size_t n, dcomplex* data, Symmetry symmetry)
{
    if (symmetry == SYMMETRY_NONE) {

        int ranks[] = { n };
        fftw_plan plan = fftw_plan_many_dft(1, ranks, howmany,
                                        reinterpret_cast<fftw_complex*>(data), NULL, howmany, 1,
                                        reinterpret_cast<fftw_complex*>(data), NULL, howmany, 1,
                                        FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);

    } else if (symmetry == SYMMETRY_EVEN) {

        int ranks[] = { n };
        fftw_r2r_kind kinds[] = { FFTW_REDFT01 };
        fftw_plan plan = fftw_plan_many_r2r(1, ranks, 2*howmany,
                                            reinterpret_cast<double*>(data), NULL, 2*howmany, 1,
                                            reinterpret_cast<double*>(data), NULL, 2*howmany, 1,
                                            kinds, FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);

    } else {

        int ranks[] = { n };
        fftw_r2r_kind kinds[] = { FFTW_RODFT01 };
        fftw_plan plan = fftw_plan_many_r2r(1, ranks, 2*howmany,
                                            reinterpret_cast<double*>(data), NULL, 2*howmany, 1,
                                            reinterpret_cast<double*>(data), NULL, 2*howmany, 1,
                                            kinds, FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);

    }
}


void FFT::backward(size_t howmany, size_t n1, size_t n2, dcomplex* data, Symmetry symmetry1, Symmetry symmetry2)
{
}

}}} // namespace plask::solvers::slab

#else

#endif
