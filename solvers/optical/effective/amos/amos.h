#ifndef PLASK__MODULE_OPTICAL_EFFECTIVE_AMOS_H
#define PLASK__MODULE_OPTICAL_EFFECTIVE_AMOS_H

#include <plask/plask.hpp>

extern "C" void F77_GLOBAL(zairy,ZAIRY)(const double& zr, const double& zi, const int& id, const int& kode,
                                        double& air, double& aii, int& nz, int& ierr);

extern "C" void F77_GLOBAL(zbiry,ZBIRY)(const double& zr, const double& zi, const int& id, const int& kode,
                                        double& bir, double& bii, int& ierr);

extern "C" void F77_GLOBAL(zbesi,ZBESI)(const double& zr, const double& zi, const double& fnu, const int& kode, const int& n,
                                        double* cyr, double* cyi, int& nz, int& ierr);

extern "C" void F77_GLOBAL(zbesj,ZBESJ)(const double& zr, const double& zi, const double& fnu, const int& kode, const int& n,
                                        double* cyr, double* cyi, int& nz, int& ierr);

extern "C" void F77_GLOBAL(zbesk,ZBESK)(const double& zr, const double& zi, const double& fnu, const int& kode, const int& n,
                                        double* cyr, double* cyi, int& nz, int& ierr);

extern "C" void F77_GLOBAL(zbesy,ZBESY)(const double& zr, const double& zi, const double& fnu, const int& kode, const int& n,
                                        double* cyr, double* cyi, int& nz, double* cwrkr, double* cwrki, int& ierr);

extern "C" void F77_GLOBAL(zbesh,ZBESH)(const double& zr, const double& zi, const double& fnu, const int& kode, const int& m, const int& n,
                                        double* cyr, double* cyi, int& nz, int&);



namespace plask {



} // namespace plask

#endif // PLASK__MODULE_OPTICAL_EFFECTIVE_AMOS_H