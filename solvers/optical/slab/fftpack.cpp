#include "fft.h"

#ifndef USE_FFTW // use fftpacx instead of fftw

#include <fftpacx/fftpacx.h>

#define lensav(n) 2*n + int(log2(n)) + 6

namespace plask { namespace solvers { namespace slab { namespace FFT {

Forward1D::Forward1D(): wsave(nullptr) {}

Forward1D::Forward1D(Forward1D&& old):
    lot(old.lot), n(old.n), strid(old.strid),
    symmetry(old.symmetry),
    wsave(old.wsave) {
    old.wsave = nullptr;
}

Forward1D& Forward1D::operator=(Forward1D&& old) {
    lot = old.lot; n = old.n; strid = old.strid;
    symmetry = old.symmetry;
    aligned_free(wsave);
    wsave = old.wsave;
    old.wsave = nullptr;
    return *this;
}

Forward1D::Forward1D(int lot, int n, Symmetry symmetry, int strid):
    lot(lot), n(n), strid(strid?strid:lot), symmetry(symmetry), wsave(aligned_malloc<double>(lensav(n))) {
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
            cfftmf_(lot, 1, n, strid, data, strid*n, wsave, lensav(n), work, 2*lot*n, ier);
        } else {
            cosqmb_(2*lot, 1, n, 2*strid, (double*)data, 2*strid*n, wsave, lensav(n), work, 2*lot*n, ier);
            double factor = 1./n;
            for (int i = 0, N = strid*n; i < N; i += strid)
                for (int j = 0; j < lot; ++j)
                    data[i+j] *= factor;
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
    lot(old.lot), n(old.n), strid(old.strid),
    symmetry(old.symmetry),
    wsave(old.wsave) {
    old.wsave = nullptr;
}

Backward1D& Backward1D::operator=(Backward1D&& old) {
    lot = old.lot; n = old.n; strid = old.strid;
    symmetry = old.symmetry;
    aligned_free(wsave);
    wsave = old.wsave;
    old.wsave = nullptr;
    return *this;
}

Backward1D::Backward1D(int lot, int n, Symmetry symmetry, int strid):
    lot(lot), n(n), strid(strid?strid:lot), symmetry(symmetry), wsave(aligned_malloc<double>(lensav(n))) {
    try {
        int ier;
        switch (symmetry) {
            case SYMMETRY_NONE:
                cfftmi_(n, wsave, lensav(n), ier); return;
            case SYMMETRY_EVEN:
                cosqmi_(n, wsave, lensav(n), ier); return;
            case SYMMETRY_ODD:
                sinqmi_(n, wsave, lensav(n), ier); return;
        }
    } catch (const std::string& msg) {
        throw CriticalException("FFT::Backward1D::Backward1D: %1%", msg);
    }
}

void Backward1D::execute(dcomplex* data) {
    if (!wsave) throw CriticalException("FFTPACX not initialized");
    try {
        int ier;
        double work[2*lot*n];
        switch (symmetry) {
            case SYMMETRY_NONE:
                cfftmb_(lot, 1, n, strid, data, strid*n, wsave, lensav(n), work, 2*lot*n, ier);
                return;
            case SYMMETRY_EVEN:
                cosqmf_(2*lot, 1, n, 2*strid, (double*)data, 2*strid*n, wsave, lensav(n), work, 2*lot*n, ier);
                break;
            case SYMMETRY_ODD:
                sinqmf_(2*lot, 1, n, 2*strid, (double*)data, 2*strid*n, wsave, lensav(n), work, 2*lot*n, ier);
                break;
        }
        double factor = n;
        for (int i = 0, N = strid*n; i < N; i += strid)
            for (int j = 0; j < lot; ++j)
                data[i+j] *= factor;
    } catch (const std::string& msg) {
        throw CriticalException("FFT::Backward1D::execute: %1%", msg);
    }
}

Backward1D::~Backward1D() {
    aligned_free(wsave);
}


Forward2D::Forward2D(): wsave1(nullptr), wsave2(nullptr) {}

Forward2D::Forward2D(Forward2D&& old):
    lot(old.lot), n1(old.n1), n2(old.n2), strid(old.strid),
    symmetry1(old.symmetry1), symmetry2(old.symmetry2),
    wsave1(old.wsave1), wsave2(old.wsave2) {
    old.wsave1 = nullptr; old.wsave2 = nullptr;
}

Forward2D& Forward2D::operator=(Forward2D&& old) {
    lot = old.lot; n1 = old.n1; n2 = old.n2; strid = old.strid;
    symmetry1 = old.symmetry1; symmetry2 = old.symmetry2;
    aligned_free(wsave1); aligned_free(wsave2);
    wsave1 = old.wsave1; wsave2 = old.wsave2;
    old.wsave1 = nullptr;  old.wsave2 = nullptr;
    return *this;
}

Forward2D::Forward2D(int lot, int n1, int n2, Symmetry symmetry1, Symmetry symmetry2, int strid):
    lot(lot), n1(n1), n2(n2), strid(strid?strid:lot), symmetry1(symmetry1), symmetry2(symmetry2),
    wsave1(aligned_malloc<double>(lensav(n1))) {
    if (n1 == n2 && symmetry1 == symmetry2) wsave2 = wsave1;
    else wsave2 = aligned_malloc<double>(lensav(n2));
    try {
        int ier;
        if (symmetry1 == SYMMETRY_NONE)
            cfftmi_(n1, wsave1, lensav(n1), ier);
        else if (symmetry1 == SYMMETRY_EVEN)
            cosqmi_(n1, wsave1, lensav(n1), ier);
        else
            throw NotImplemented("forward FFT for odd symmetry");
        if (wsave1 != wsave2) {
            if (symmetry2 == SYMMETRY_NONE)
                cfftmi_(n2, wsave2, lensav(n2), ier);
            else if (symmetry2 == SYMMETRY_EVEN)
                cosqmi_(n2, wsave2, lensav(n2), ier);
            else
                throw NotImplemented("forward FFT for odd symmetry");
        }
    } catch (const std::string& msg) {
        throw CriticalException("FFT::Forward2D::Forward2D: %1%", msg);
    }
}

void Forward2D::execute(dcomplex* data) {
    if (!wsave1 || !wsave2) throw CriticalException("FFTPACX not initialized");
    try {
        int ier;
        double work[2*lot*max(n1,n2)];
        // n1 is changing faster than n2
        if (symmetry1 == SYMMETRY_NONE) {
            for (int i = 0; i != n2; ++i)
                cfftmf_(lot, 1, n1, strid, data+strid*n1*i, strid*n1, wsave1, lensav(n1), work, 2*lot*n1, ier);
        } else {
            for (int i = 0; i != n2; ++i)
                cosqmb_(2*lot, 1, n1, 2*strid, (double*)data+2*strid*n1*i, 2*strid*n1, wsave1, lensav(n1), work, 2*lot*n1, ier);
            double factor = 1./n1;
            for (int i = 0, N = strid*n1*n2; i < N; i += strid)
                for (int j = 0; j < lot; ++j)
                    data[i+j] *= factor;
        }
        if (symmetry2 == SYMMETRY_NONE) {
            for (int i = 0; i != n1; ++i)
                cfftmf_(lot, 1, n2, strid*n1, data+strid*i, strid*n1*n2, wsave2, lensav(n2), work, 2*lot*n2, ier);
        } else {
            for (int i = 0; i != n1; ++i)
                cosqmb_(2*lot, 1, n1, 2*strid*n1, (double*)data+2*strid*i, 2*strid*(n1*(n2-1)+1), wsave2, lensav(n2), work, 2*lot*n2, ier);
            double factor = 1./n2;
            for (int i = 0, N = strid*n1*n2; i < N; i += strid)
                for (int j = 0; j < lot; ++j)
                    data[i+j] *= factor;
        }
    } catch (const std::string& msg) {
        throw CriticalException("FFT::Forward2D::execute: %1%", msg);
    }
}

Forward2D::~Forward2D() {
    if (wsave2 != wsave1) aligned_free(wsave2);
    aligned_free(wsave1);
}










}}}} // namespace plask::solvers::slab

#endif // USE_FFTW

