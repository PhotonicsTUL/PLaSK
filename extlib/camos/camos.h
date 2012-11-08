#ifndef AMOS_H
#define AMOS_H

#include <plask/plask.hpp>

extern "C" void zairy(const double& zr, const double& zi, const long& id, const long& kode,
                      double& air, double& aii, long& nz, long& ierr);

extern "C" void zbiry(const double& zr, const double& zi, const long& id, const long& kode,
                      double& bir, double& bii, long& ierr);

extern "C" void zbesi(const double& zr, const double& zi, const double& fnu, const long& kode, const long& n,
                      double* cyr, double* cyi, long& nz, long& ierr);

extern "C" void zbesj(const double& zr, const double& zi, const double& fnu, const long& kode, const long& n,
                      double* cyr, double* cyi, long& nz, long& ierr);

extern "C" void zbesk(const double& zr, const double& zi, const double& fnu, const long& kode, const long& n,
                      double* cyr, double* cyi, long& nz, long& ierr);

extern "C" void zbesy(const double& zr, const double& zi, const double& fnu, const long& kode, const long& n,
                      double* cyr, double* cyi, long& nz, double* cwrkr, double* cwrki, long& ierr);

extern "C" void zbesh(const double& zr, const double& zi, const double& fnu, const long& kode, const long& m, const long& n,
                      double* cyr, double* cyi, long& nz, long&);

#endif // AMOS_H