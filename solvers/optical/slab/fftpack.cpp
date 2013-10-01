#include "fft.h"

#ifndef USE_FFTW // use fftpacx instead of fftw

#include <fftpacx/fftpacx.h>

#define lensav(n) 2*n + int(log2(n)) + 6

namespace plask { namespace solvers { namespace slab { namespace FFT {

Forward1D::Forward1D(): wsave(nullptr) {}

Forward1D::Forward1D(Forward1D&& old):
    lot(old.lot), n(old.n), st(old.st),
    symmetry(old.symmetry),
    wsave(old.wsave) {
    old.wsave = nullptr;
}

Forward1D& Forward1D::operator=(Forward1D&& old) {
    lot = old.lot; n = old.n; st = old.st;
    symmetry = old.symmetry;
    wsave = old.wsave;
    old.wsave = nullptr;
    return *this;
}

Forward1D::Forward1D(int lot, int n, Symmetry symmetry, int st):
    lot(lot), n(n), st(st), symmetry(symmetry), wsave(aligned_malloc<double>(lensav(n))) {
    try { 
        int ier;
        if (symmetry == SYMMETRY_NONE)
            cfftmi_(n, wsave, lensav(n), ier);
        else if (symmetry == SYMMETRY_EVEN)
            cosqmi_(n, wsave, lensav(n), ier);
        else
            throw NotImplemented("forward FFT for odd symmetry");
    } catch (const std::string& msg) {
        throw CriticalException("FFT::Forward1D::Forward1D: %1%", msg);
    }
}

void Forward1D::execute(dcomplex* data) {
    if (!wsave) throw CriticalException("FFTPACX not initialized");
    try { 
        int ier;
        double work[2*lot*n];
        if (symmetry == SYMMETRY_NONE) {
            cfftmf_(lot, st, n, lot*st, data, lot*n, wsave, lensav(n), work, 2*lot*n, ier);
        } else {
            cosqmb_(2*lot, st, n, 2*lot*st, (double*)data, 2*lot*n, wsave, lensav(n), work, 2*lot*n, ier);
            double factor = 1./n; for (int N = lot*n, i = 0; i < N; ++i) data[i] *= factor;
        }
    } catch (const std::string& msg) {
        throw CriticalException("FFT::Forward1D::execute: %1%", msg);
    }
}

Forward1D::~Forward1D() {
    aligned_free(wsave);
}




Backward1D::Backward1D(): wsave(nullptr) {}

Backward1D::Backward1D(Backward1D&& old):
    lot(old.lot), n(old.n), st(old.st),
    symmetry(old.symmetry),
    wsave(old.wsave) {
    old.wsave = nullptr;
}

Backward1D& Backward1D::operator=(Backward1D&& old) {
    lot = old.lot; n = old.n; st = old.st;
    symmetry = old.symmetry;
    wsave = old.wsave;
    old.wsave = nullptr;
    return *this;
}

Backward1D::Backward1D(int lot, int n, Symmetry symmetry, int st):
    lot(lot), n(n), st(st), symmetry(symmetry), wsave(aligned_malloc<double>(lensav(n))) {
    try { 
        int ier;
        if (symmetry == SYMMETRY_NONE)
            cfftmi_(n, wsave, lensav(n), ier);
        else if (symmetry == SYMMETRY_EVEN)
            cosqmi_(n, wsave, lensav(n), ier);
        else
            sinqmi_(n, wsave, lensav(n), ier);
    } catch (const std::string& msg) {
        throw CriticalException("FFT::Backward1D::Backward1D: %1%", msg);
    }
}

void Backward1D::execute(dcomplex* data) {
    if (!wsave) throw CriticalException("FFTPACX not initialized");
    try { 
        int ier;
        double work[2*lot*n];
        if (symmetry == SYMMETRY_NONE) {
            cfftmb_(lot, st, n, lot*st, data, lot*n, wsave, lensav(n), work, 2*lot*n, ier);
        } else if (symmetry == SYMMETRY_EVEN) {
            cosqmf_(2*lot, st, n, 2*lot*st, (double*)data, 2*lot*n, wsave, lensav(n), work, 2*lot*n, ier);
             double factor = n; for (int N = lot*n, i = 0; i < N; ++i) data[i] *= factor;
        } else if (symmetry == SYMMETRY_EVEN) {
            sinqmf_(2*lot, st, n, 2*lot*st, (double*)data, 2*lot*n, wsave, lensav(n), work, 2*lot*n, ier);
            double factor = n; for (int N = lot*n, i = 0; i < N; ++i) data[i] *= factor;
         }
    } catch (const std::string& msg) {
        throw CriticalException("FFT::Backward1D::execute: %1%", msg);
    }
}

Backward1D::~Backward1D() {
    aligned_free(wsave);
}











}}}} // namespace plask::solvers::slab

#endif // USE_FFTW

