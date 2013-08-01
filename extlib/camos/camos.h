#ifndef AMOS_H
#define AMOS_H

extern "C" void zairy(const double& zr, const double& zi, const long& id, const long& kode,
                      double& air, double& aii, long& nz, long& ierr);

extern "C" void zbiry(const double& zr, const double& zi, const long& id, const long& kode,
                      double& bir, double& bii, long& ierr);

/**
 * Compute Bessel I function of a complex argument.
 *
 * On kode=1, zbesj computes an n member  sequence of complex
 * Bessel functions cy(i)=I(fnu+i-1,z) for real, nonnegative
 * orders fnu+i-1, i=1,...,n and complex z in the cut plane
 * -pi < arg(z) < pi. On kode=2, cbesj returns the scaled
 * functions cy(i)=exp(-abs(y))*I(fnu+i-1,z)   i = 1,...,n , y=aimag(z)
 * which remove the exponential growth in both the upper and
 * lower half planes for z to infinity. Definitions and notation
 * are found in the nbs handbook of mathematical functions.
 * \param[in] zr,zi  z=cmplx(zr,zi),  -pi < arg(z) <= pi
 * \param[in] fnu    order of initial j function, fnu >= 0.0d0
 * \param[in] kode   a parameter to indicate the scaling option
 *                   kode = 1  returns
 *                             cy(i)=j(fnu+i-1,z), i=1,...,n
 *                        = 2  returns
 *                             cy(i)=j(fnu+i-1,z)exp(-abs(y)), i=1,...,n
 * \param[in] n      number of members of the sequence, n >= 1
 * \param[out] cyr,cyi double precision vectors whose first n components
 *                     contain real and imaginary parts for the sequence
 *                     cy(i)=I(fnu+i-1,z)  or
 *                     cy(i)=I(fnu+i-1,z)exp(-abs(y))  i=1,...,n
 *                     depending on kode, y=aimag(z).
 * \param[out] nz      number of components set to zero due to underflow,
 *                     nz = 0   normal return
 *                     nz > 0   last nz components of cy set  zero due
 *                              to underflow, cy(i)=cmplx(0.0d0,0.0d0),
 *                              i = n-nz+1,...,n
 * \param[out] ierr   error flag
 *                    ierr = 0 normal return - computation completed
 *                    ierr = 1 input error   - no computation
 *                    ierr = 2 overflow      - no computation, aimag(z)
 *                             too large on kode=1
 *                    ierr = 3 cabs(z) or fnu+n-1 large - computation done
 *                             but losses of signifcance by argument
 *                             reduction produce less than half of machine
 *                             accuracy
 *                    ierr = 4 cabs(z) or fnu+n-1 too large - no computation
 *                             because of complete losses of significance
 *                             by argument reduction
 *                    ierr = 5 error              - no computation,
 *                             algorithm termination condition not met
 */
extern "C" void zbesi(const double& zr, const double& zi, const double& fnu, const long& kode, const long& n,
                      double* cyr, double* cyi, long& nz, long& ierr);

/**
 * Compute Bessel J function of a complex argument.
 *
 * On kode=1, zbesj computes an n member  sequence of complex
 * Bessel functions cy(i)=J(fnu+i-1,z) for real, nonnegative
 * orders fnu+i-1, i=1,...,n and complex z in the cut plane
 * -pi < arg(z) < pi. On kode=2, cbesj returns the scaled
 * functions cy(i)=exp(-abs(y))*J(fnu+i-1,z)   i = 1,...,n , y=aimag(z)
 * which remove the exponential growth in both the upper and
 * lower half planes for z to infinity. Definitions and notation
 * are found in the nbs handbook of mathematical functions.
 * \param[in] zr,zi  z=cmplx(zr,zi),  -pi < arg(z) <= pi
 * \param[in] fnu    order of initial j function, fnu >= 0.0d0
 * \param[in] kode   a parameter to indicate the scaling option
 *                   kode = 1  returns
 *                             cy(i)=J(fnu+i-1,z), i=1,...,n
 *                        = 2  returns
 *                             cy(i)=J(fnu+i-1,z)exp(-abs(y)), i=1,...,n
 * \param[in] n      number of members of the sequence, n >= 1
 * \param[out] cyr,cyi double precision vectors whose first n components
 *                     contain real and imaginary parts for the sequence
 *                     cy(i)=J(fnu+i-1,z)  or
 *                     cy(i)=J(fnu+i-1,z)exp(-abs(y))  i=1,...,n
 *                     depending on kode, y=aimag(z).
 * \param[out] nz      number of components set to zero due to underflow,
 *                     nz = 0   normal return
 *                     nz > 0   last nz components of cy set  zero due
 *                              to underflow, cy(i)=cmplx(0.0d0,0.0d0),
 *                              i = n-nz+1,...,n
 * \param[out] ierr   error flag
 *                    ierr = 0 normal return - computation completed
 *                    ierr = 1 input error   - no computation
 *                    ierr = 2 overflow      - no computation, aimag(z)
 *                             too large on kode=1
 *                    ierr = 3 cabs(z) or fnu+n-1 large - computation done
 *                             but losses of signifcance by argument
 *                             reduction produce less than half of machine
 *                             accuracy
 *                    ierr = 4 cabs(z) or fnu+n-1 too large - no computation
 *                             because of complete losses of significance
 *                             by argument reduction
 *                    ierr = 5 error              - no computation,
 *                             algorithm termination condition not met
 */
extern "C" void zbesj(const double& zr, const double& zi, const double& fnu, const long& kode, const long& n,
                      double* cyr, double* cyi, long& nz, long& ierr);

/**
 * Compute Bessel K function of a complex argument.
 *
 * On kode=1, zbesj computes an n member  sequence of complex
 * Bessel functions cy(i)=K(fnu+i-1,z) for real, nonnegative
 * orders fnu+i-1, i=1,...,n and complex z in the cut plane
 * -pi < arg(z) < pi. On kode=2, cbesj returns the scaled
 * functions cy(i)=exp(-abs(y))*K(fnu+i-1,z)   i = 1,...,n , y=aimag(z)
 * which remove the exponential growth in both the upper and
 * lower half planes for z to infinity. Definitions and notation
 * are found in the nbs handbook of mathematical functions.
 * \param[in] zr,zi  z=cmplx(zr,zi),  -pi < arg(z) <= pi
 * \param[in] fnu    order of initial j function, fnu >= 0.0d0
 * \param[in] kode   a parameter to indicate the scaling option
 *                   kode = 1  returns
 *                             cy(i)=K(fnu+i-1,z), i=1,...,n
 *                        = 2  returns
 *                             cy(i)=K(fnu+i-1,z)exp(-abs(y)), i=1,...,n
 * \param[in] n      number of members of the sequence, n >= 1
 * \param[out] cyr,cyi double precision vectors whose first n components
 *                     contain real and imaginary parts for the sequence
 *                     cy(i)=K(fnu+i-1,z)  or
 *                     cy(i)=K(fnu+i-1,z)exp(-abs(y))  i=1,...,n
 *                     depending on kode, y=aimag(z).
 * \param[out] nz      number of components set to zero due to underflow,
 *                     nz = 0   normal return
 *                     nz > 0   last nz components of cy set  zero due
 *                              to underflow, cy(i)=cmplx(0.0d0,0.0d0),
 *                              i = n-nz+1,...,n
 * \param[out] ierr   error flag
 *                    ierr = 0 normal return - computation completed
 *                    ierr = 1 input error   - no computation
 *                    ierr = 2 overflow      - no computation, aimag(z)
 *                             too large on kode=1
 *                    ierr = 3 cabs(z) or fnu+n-1 large - computation done
 *                             but losses of signifcance by argument
 *                             reduction produce less than half of machine
 *                             accuracy
 *                    ierr = 4 cabs(z) or fnu+n-1 too large - no computation
 *                             because of complete losses of significance
 *                             by argument reduction
 *                    ierr = 5 error              - no computation,
 *                             algorithm termination condition not met
 */
extern "C" void zbesk(const double& zr, const double& zi, const double& fnu, const long& kode, const long& n,
                      double* cyr, double* cyi, long& nz, long& ierr);

/**
 * Compute Bessel Y function of a complex argument.
 *
 * On kode=1, zbesj computes an n member  sequence of complex
 * Bessel functions cy(i)=Y(fnu+i-1,z) for real, nonnegative
 * orders fnu+i-1, i=1,...,n and complex z in the cut plane
 * -pi < arg(z) < pi. On kode=2, cbesj returns the scaled
 * functions cy(i)=exp(-abs(y))*Y(fnu+i-1,z)   i = 1,...,n , y=aimag(z)
 * which remove the exponential growth in both the upper and
 * lower half planes for z to infinity. Definitions and notation
 * are found in the nbs handbook of mathematical functions.
 * \param[in] zr,zi  z=cmplx(zr,zi),  -pi < arg(z) <= pi
 * \param[in] fnu    order of initial j function, fnu >= 0.0d0
 * \param[in] kode   a parameter to indicate the scaling option
 *                   kode = 1  returns
 *                             cy(i)=Y(fnu+i-1,z), i=1,...,n
 *                        = 2  returns
 *                             cy(i)=Y(fnu+i-1,z)exp(-abs(y)), i=1,...,n
 * \param[in] n      number of members of the sequence, n >= 1
 * \param[out] cyr,cyi double precision vectors whose first n components
 *                     contain real and imaginary parts for the sequence
 *                     cy(i)=Y(fnu+i-1,z)  or
 *                     cy(i)=Y(fnu+i-1,z)exp(-abs(y))  i=1,...,n
 *                     depending on kode, y=aimag(z).
 * \param[out] nz      number of components set to zero due to underflow,
 *                     nz = 0   normal return
 *                     nz > 0   last nz components of cy set  zero due
 *                              to underflow, cy(i)=cmplx(0.0d0,0.0d0),
 *                              i = n-nz+1,...,n
 * \param[out] ierr   error flag
 *                    ierr = 0 normal return - computation completed
 *                    ierr = 1 input error   - no computation
 *                    ierr = 2 overflow      - no computation, aimag(z)
 *                             too large on kode=1
 *                    ierr = 3 cabs(z) or fnu+n-1 large - computation done
 *                             but losses of signifcance by argument
 *                             reduction produce less than half of machine
 *                             accuracy
 *                    ierr = 4 cabs(z) or fnu+n-1 too large - no computation
 *                             because of complete losses of significance
 *                             by argument reduction
 *                    ierr = 5 error              - no computation,
 *                             algorithm termination condition not met
 */
extern "C" void zbesy(const double& zr, const double& zi, const double& fnu, const long& kode, const long& n,
                      double* cyr, double* cyi, long& nz, double* cwrkr, double* cwrki, long& ierr);

/**
 * Compute Bessel H function of a complex argument.
 *
 * On kode=1, zbesj computes an n member  sequence of complex
 * Bessel functions cy(i)=H(fnu+i-1,z) for real, nonnegative
 * orders fnu+i-1, i=1,...,n and complex z in the cut plane
 * -pi < arg(z) < pi. On kode=2, cbesj returns the scaled
 * functions cy(i)=exp(-abs(y))*H(fnu+i-1,z)   i = 1,...,n , y=aimag(z)
 * which remove the exponential growth in both the upper and
 * lower half planes for z to infinity. Definitions and notation
 * are found in the nbs handbook of mathematical functions.
 * \param[in] zr,zi  z=cmplx(zr,zi),  -pi < arg(z) <= pi
 * \param[in] fnu    order of initial j function, fnu >= 0.0d0
 * \param[in] kode   a parameter to indicate the scaling option
 *                   kode = 1  returns
 *                             cy(i)=H(fnu+i-1,z), i=1,...,n
 *                        = 2  returns
 *                             cy(i)=H(fnu+i-1,z)exp(-abs(y)), i=1,...,n
 * \param[in] n      number of members of the sequence, n >= 1
 * \param[in] m      type of the Hankel function
 *                   m = 1  H(z) = J(z) + j H(z)
 *                   m = 2  H(z) = J(z) - j H(z)
 * \param[out] cyr,cyi double precision vectors whose first n components
 *                     contain real and imaginary parts for the sequence
 *                     cy(i)=H(fnu+i-1,z)  or
 *                     cy(i)=H(fnu+i-1,z)exp(-abs(y))  i=1,...,n
 *                     depending on kode, y=aimag(z).
 * \param[out] nz      number of components set to zero due to underflow,
 *                     nz = 0   normal return
 *                     nz > 0   last nz components of cy set  zero due
 *                              to underflow, cy(i)=cmplx(0.0d0,0.0d0),
 *                              i = n-nz+1,...,n
 * \param[out] ierr   error flag
 *                    ierr = 0 normal return - computation completed
 *                    ierr = 1 input error   - no computation
 *                    ierr = 2 overflow      - no computation, aimag(z)
 *                             too large on kode=1
 *                    ierr = 3 cabs(z) or fnu+n-1 large - computation done
 *                             but losses of signifcance by argument
 *                             reduction produce less than half of machine
 *                             accuracy
 *                    ierr = 4 cabs(z) or fnu+n-1 too large - no computation
 *                             because of complete losses of significance
 *                             by argument reduction
 *                    ierr = 5 error              - no computation,
 *                             algorithm termination condition not met
 */
extern "C" void zbesh(const double& zr, const double& zi, const double& fnu, const long& kode, const long& m, const long& n,
                      double* cyr, double* cyi, long& nz, long&);

#endif // AMOS_H