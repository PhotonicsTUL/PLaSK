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
        switch (symmetry) {
            case (SYMMETRY_NONE):
                cfftmi_(n, wsave, lensav(n), ier); return;
            case (SYMMETRY_EVEN_2):
                cosqmi_(n, wsave, lensav(n), ier); return;
            case (SYMMETRY_EVEN_1):
                costmi_(n, wsave, lensav(n), ier); return;
            default:
                throw NotImplemented("forward FFT for odd symmetry");
        }
    } catch (const std::string& msg) {
        throw CriticalException("FFT::Forward1D::Forward1D: %1%", msg);
    }
}

void Forward1D::execute(dcomplex* data) {
    if (!wsave) throw CriticalException("FFTPACX not initialized");
    try {
        int ier;
        double work[2*lot*n];
        switch (symmetry) {
            case (SYMMETRY_NONE):
                cfftmf_(lot, 1, n, strid, data, strid*n, wsave, lensav(n), work, 2*lot*n, ier);
                return;
            case (SYMMETRY_EVEN_2):
                cosqmb_(2*lot, 1, n, 2*strid, (double*)data, 2*strid*n, wsave, lensav(n), work, 2*lot*n, ier);
                break;
            case (SYMMETRY_EVEN_1):
                costmb_(2*lot, 1, n, 2*strid, (double*)data, 2*strid*n, wsave, lensav(n), work, 2*lot*n, ier);
                break;
            default: {} // silence the warning
        }
        double factor = 1./n;
        for (int i = 0, N = strid*n; i < N; i += strid)
            for (int j = 0; j < lot; ++j)
                data[i+j] *= factor;
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
            case SYMMETRY_EVEN_2:
                cosqmi_(n, wsave, lensav(n), ier); return;
            case SYMMETRY_ODD_2:
                sinqmi_(n, wsave, lensav(n), ier); return;
            case (SYMMETRY_EVEN_1):
                costmi_(n, wsave, lensav(n), ier); return;
            case (SYMMETRY_ODD_1):
                sintmi_(n, wsave, lensav(n), ier); return;
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
            case SYMMETRY_EVEN_2:
                cosqmf_(2*lot, 1, n, 2*strid, (double*)data, 2*strid*n, wsave, lensav(n), work, 2*lot*n, ier);
                break;
            case SYMMETRY_ODD_2:
                sinqmf_(2*lot, 1, n, 2*strid, (double*)data, 2*strid*n, wsave, lensav(n), work, 2*lot*n, ier);
                break;
            case SYMMETRY_EVEN_1:
                costmf_(2*lot, 1, n, 2*strid, (double*)data, 2*strid*n, wsave, lensav(n), work, 2*lot*n, ier);
                break;
            case SYMMETRY_ODD_1:
                sintmf_(2*lot, 1, n, 2*strid, (double*)data, 2*strid*n, wsave, lensav(n), work, 2*lot*n, ier);
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
    lot(old.lot), n1(old.n1), n2(old.n2),
    strid(old.strid), strid2(old.strid2),
    symmetry1(old.symmetry1), symmetry2(old.symmetry2),
    wsave1(old.wsave1), wsave2(old.wsave2) {
    old.wsave1 = nullptr; if (old.wsave2 != old.wsave1) old.wsave2 = nullptr;
}

Forward2D& Forward2D::operator=(Forward2D&& old) {
    lot = old.lot; n1 = old.n1; n2 = old.n2;
    strid = old.strid; strid2 = old.strid2;
    symmetry1 = old.symmetry1; symmetry2 = old.symmetry2;
    aligned_free(wsave1); if (wsave2 != wsave1) aligned_free(wsave2);
    wsave1 = old.wsave1; wsave2 = old.wsave2;
    old.wsave1 = nullptr; if (old.wsave2 != old.wsave1) old.wsave2 = nullptr;
    return *this;
}

Forward2D::Forward2D(int lot, int n1, int n2, Symmetry symmetry1, Symmetry symmetry2, int strid, int ld):
    lot(lot), n1(n1), n2(n2), strid(strid?strid:lot), strid2((strid?strid:lot)*(ld?ld:n1)), symmetry1(symmetry1), symmetry2(symmetry2),
    wsave1(aligned_malloc<double>(lensav(n1))) {
    if (n1 == n2 && symmetry1 == symmetry2) wsave2 = wsave1;
    else wsave2 = aligned_malloc<double>(lensav(n2));
    try {
        int ier;
        switch (symmetry1) {
            case SYMMETRY_NONE:
                cfftmi_(n1, wsave1, lensav(n1), ier); return;
            case SYMMETRY_EVEN_2:
                cosqmi_(n1, wsave1, lensav(n1), ier); return;
            case (SYMMETRY_EVEN_1):
                costmi_(n1, wsave1, lensav(n1), ier); return;
            default:
                throw NotImplemented("forward FFT for odd symmetry");
        }
        if (wsave1 != wsave2) {
            switch (symmetry2) {
                case SYMMETRY_NONE:
                    cfftmi_(n2, wsave2, lensav(n2), ier); return;
                case SYMMETRY_EVEN_2:
                    cosqmi_(n2, wsave2, lensav(n2), ier); return;
                case (SYMMETRY_EVEN_1):
                    costmi_(n2, wsave2, lensav(n2), ier); return;
                default:
                    throw NotImplemented("forward FFT for odd symmetry");
            }
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
        double factor1 = 1./n1;
        switch (symmetry1) {
            case (SYMMETRY_NONE):
                for (int i = 0; i != n2; ++i)
                    cfftmf_(lot, 1, n1, strid, data+strid2*i, strid2, wsave1, lensav(n1), work, 2*lot*n1, ier);
                break;
            case (SYMMETRY_EVEN_2):
                for (int i = 0; i != n2; ++i) {
                    cosqmb_(2*lot, 1, n1, 2*strid, (double*)data+2*strid2*i, 2*strid2, wsave1, lensav(n1), work, 2*lot*n1, ier);
                    for (int j = 0, shift = strid2*i, N = strid*n1; j < N; j += strid)
                        for (int l = 0; l < lot; ++l)
                            data[shift+j+l] *= factor1;
                }
                break;
            case (SYMMETRY_EVEN_1):
                for (int i = 0; i != n2; ++i) {
                    costmb_(2*lot, 1, n1, 2*strid, (double*)data+2*strid2*i, 2*strid2, wsave1, lensav(n1), work, 2*lot*n1, ier);
                    for (int j = 0, shift = strid2*i, N = strid*n1; j < N; j += strid)
                        for (int l = 0; l < lot; ++l)
                            data[shift+j+l] *= factor1;
                }
                break;
            default: {} // silence the warning
        }
        double factor2 = 1./n2;
        switch (symmetry2) {
            case (SYMMETRY_NONE):
                for (int i = 0; i != n1; ++i)
                    cfftmf_(lot, 1, n2, strid2, data+strid*i, strid+strid2*(n2-1), wsave2, lensav(n2), work, 2*lot*n2, ier);
                break;
            case (SYMMETRY_EVEN_2):
                for (int i = 0; i != n1; ++i) {
                    cosqmb_(2*lot, 1, n2, 2*strid2, (double*)data+2*strid*i, 2*(strid+strid2*(n2-1)), wsave2, lensav(n2), work, 2*lot*n2, ier);
                    for (int j = 0, shift = strid*i, N = n2*strid2; j < N; j += strid2)
                        for (int l = 0; l < lot; ++l)
                            data[shift+j+l] *= factor2;
                }
                break;
            case (SYMMETRY_EVEN_1):
                for (int i = 0; i != n1; ++i) {
                    costmb_(2*lot, 1, n2, 2*strid2, (double*)data+2*strid*i, 2*(strid+strid2*(n2-1)), wsave2, lensav(n2), work, 2*lot*n2, ier);
                    for (int j = 0, shift = strid*i, N = n2*strid2; j < N; j += strid2)
                        for (int l = 0; l < lot; ++l)
                            data[shift+j+l] *= factor2;
                }
                break;
            default: {} // silence the warning
        }
    } catch (const std::string& msg) {
        throw CriticalException("FFT::Forward2D::execute: %1%", msg);
    }
}

Forward2D::~Forward2D() {
    if (wsave2 != wsave1) aligned_free(wsave2);
    aligned_free(wsave1);
}



Backward2D::Backward2D(): wsave1(nullptr), wsave2(nullptr) {}

Backward2D::Backward2D(Backward2D&& old):
    lot(old.lot), n1(old.n1), n2(old.n2),
    strid(old.strid), strid2(old.strid2),
    symmetry1(old.symmetry1), symmetry2(old.symmetry2),
    wsave1(old.wsave1), wsave2(old.wsave2) {
    old.wsave1 = nullptr; if (old.wsave2 != old.wsave1) old.wsave2 = nullptr;
}

Backward2D& Backward2D::operator=(Backward2D&& old) {
    lot = old.lot; n1 = old.n1; n2 = old.n2;
    strid = old.strid; strid2 = old.strid2;
    symmetry1 = old.symmetry1; symmetry2 = old.symmetry2;
    aligned_free(wsave1); if (wsave2 != wsave1) aligned_free(wsave2);
    wsave1 = old.wsave1; wsave2 = old.wsave2;
    old.wsave1 = nullptr; if (old.wsave2 != old.wsave1) old.wsave2 = nullptr;
    return *this;
}

Backward2D::Backward2D(int lot, int n1, int n2, Symmetry symmetry1, Symmetry symmetry2, int strid, int ld):
    lot(lot), n1(n1), n2(n2), strid(strid?strid:lot), strid2((strid?strid:lot)*(ld?ld:n1)), symmetry1(symmetry1), symmetry2(symmetry2),
    wsave1(aligned_malloc<double>(lensav(n1))) {
    if (n1 == n2 && symmetry1 == symmetry2) wsave2 = wsave1;
    else wsave2 = aligned_malloc<double>(lensav(n2));
    try {
        int ier;
        switch (symmetry1) {
            case SYMMETRY_NONE:
                cfftmi_(n1, wsave1, lensav(n1), ier); return;
            case SYMMETRY_EVEN_2:
                cosqmi_(n1, wsave1, lensav(n1), ier); return;
            case SYMMETRY_ODD_2:
                sinqmi_(n1, wsave1, lensav(n1), ier); return;
            case (SYMMETRY_EVEN_1):
                costmi_(n1, wsave1, lensav(n1), ier); return;
            case (SYMMETRY_ODD_1):
                sintmi_(n1, wsave1, lensav(n1), ier); return;
        }
        if (wsave1 != wsave2) {
            switch (symmetry2) {
                case SYMMETRY_NONE:
                    cfftmi_(n2, wsave2, lensav(n2), ier); return;
                case SYMMETRY_EVEN_2:
                    cosqmi_(n2, wsave2, lensav(n2), ier); return;
                case SYMMETRY_ODD_2:
                    sinqmi_(n2, wsave2, lensav(n2), ier); return;
                case (SYMMETRY_EVEN_1):
                    costmi_(n2, wsave2, lensav(n2), ier); return;
                case (SYMMETRY_ODD_1):
                    sintmi_(n2, wsave2, lensav(n2), ier); return;
            }
        }
    } catch (const std::string& msg) {
        throw CriticalException("FFT::Backward2D::Backward2D: %1%", msg);
    }
}

void Backward2D::execute(dcomplex* data) {
    if (!wsave1 || !wsave2) throw CriticalException("FFTPACX not initialized");
    try {
        int ier;
        double work[2*lot*max(n1,n2)];
        // n1 is changing faster than n2
        double factor1 = n1;
        switch (symmetry1) {
            case (SYMMETRY_NONE):
                for (int i = 0; i != n2; ++i)
                    cfftmb_(lot, 1, n1, strid, data+strid2*i, strid2, wsave1, lensav(n1), work, 2*lot*n1, ier);
                break;
            case (SYMMETRY_EVEN_2):
                for (int i = 0; i != n2; ++i) {
                    cosqmf_(2*lot, 1, n1, 2*strid, (double*)data+2*strid2*i, 2*strid2, wsave1, lensav(n1), work, 2*lot*n1, ier);
                    for (int j = 0, shift = strid2*i, N = strid*n1; j < N; j += strid)
                        for (int l = 0; l < lot; ++l)
                            data[j+l+shift] *= factor1;
                }
                break;
            case (SYMMETRY_ODD_2):
                for (int i = 0; i != n2; ++i) {
                    sinqmf_(2*lot, 1, n1, 2*strid, (double*)data+2*strid2*i, 2*strid2, wsave1, lensav(n1), work, 2*lot*n1, ier);
                    for (int j = 0, shift = strid2*i, N = strid*n1; j < N; j += strid)
                        for (int l = 0; l < lot; ++l)
                            data[j+l+shift] *= factor1;
                }
                break;
            case (SYMMETRY_EVEN_1):
                for (int i = 0; i != n2; ++i) {
                    costmf_(2*lot, 1, n1, 2*strid, (double*)data+2*strid2*i, 2*strid2, wsave1, lensav(n1), work, 2*lot*n1, ier);
                    for (int j = 0, shift = strid2*i, N = strid*n1; j < N; j += strid)
                        for (int l = 0; l < lot; ++l)
                            data[j+l+shift] *= factor1;
                }
                break;
            case (SYMMETRY_ODD_1):
                for (int i = 0; i != n2; ++i) {
                    sintmf_(2*lot, 1, n1, 2*strid, (double*)data+2*strid2*i, 2*strid2, wsave1, lensav(n1), work, 2*lot*n1, ier);
                    for (int j = 0, shift = strid2*i, N = strid*n1; j < N; j += strid)
                        for (int l = 0; l < lot; ++l)
                            data[j+l+shift] *= factor1;
                }
                break;
        }
        double factor2 = n2;
        switch (symmetry2) {
            case (SYMMETRY_NONE):
                for (int i = 0; i != n1; ++i)
                    cfftmb_(lot, 1, n2, strid2, data+strid*i, strid+strid2*(n2-1), wsave2, lensav(n2), work, 2*lot*n2, ier);
                break;
            case (SYMMETRY_EVEN_2):
                for (int i = 0; i != n1; ++i) {
                    cosqmf_(2*lot, 1, n2, 2*strid2, (double*)data+2*strid*i, 2*(strid+strid2*(n2-1)), wsave2, lensav(n2), work, 2*lot*n2, ier);
                    for (int j = 0, shift = strid*i, N = n2*strid2; j < N; j += strid2)
                        for (int l = 0; l < lot; ++l)
                            data[shift+j+l] *= factor2;
                }
                break;
            case (SYMMETRY_ODD_2):
                for (int i = 0; i != n1; ++i) {
                    sinqmf_(2*lot, 1, n2, 2*strid2, (double*)data+2*strid*i, 2*(strid+strid2*(n2-1)), wsave2, lensav(n2), work, 2*lot*n2, ier);
                    for (int j = 0, shift = strid*i, N = n2*strid2; j < N; j += strid2)
                        for (int l = 0; l < lot; ++l)
                            data[shift+j+l] *= factor2;
                }
                break;
            case (SYMMETRY_EVEN_1):
                for (int i = 0; i != n1; ++i) {
                    costmf_(2*lot, 1, n2, 2*strid2, (double*)data+2*strid*i, 2*(strid+strid2*(n2-1)), wsave2, lensav(n2), work, 2*lot*n2, ier);
                    for (int j = 0, shift = strid*i, N = n2*strid2; j < N; j += strid2)
                        for (int l = 0; l < lot; ++l)
                            data[shift+j+l] *= factor2;
                }
                break;
            case (SYMMETRY_ODD_1):
                for (int i = 0; i != n1; ++i) {
                    sintmf_(2*lot, 1, n2, 2*strid2, (double*)data+2*strid*i, 2*(strid+strid2*(n2-1)), wsave2, lensav(n2), work, 2*lot*n2, ier);
                    for (int j = 0, shift = strid*i, N = n2*strid2; j < N; j += strid2)
                        for (int l = 0; l < lot; ++l)
                            data[shift+j+l] *= factor2;
                }
                break;
        }
    } catch (const std::string& msg) {
        throw CriticalException("FFT::Backward2D::execute: %1%", msg);
    }
}

Backward2D::~Backward2D() {
    if (wsave2 != wsave1) aligned_free(wsave2);
    aligned_free(wsave1);
}









}}}} // namespace plask::solvers::slab

#endif // USE_FFTW

