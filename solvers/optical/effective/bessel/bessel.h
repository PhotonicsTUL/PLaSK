#ifndef PLASK__MODULE_OPTICAL_EFFECTIVE_BESSEL_HPP
#define PLASK__MODULE_OPTICAL_EFFECTIVE_BESSEL_HPP

#include <cmath>
#include <complex>

namespace plask { namespace solvers { namespace effective { namespace bessel {

/**
 *   Compute Bessel J and Y functions for real parameter.
 *   This algorithm computes the above functions using series expansions.
 *
 *   \param x    -- argument of Bessel function
 *
 *   \param[out] j0   -- Bessel function of 1st kind, 0th order
 *   \param[out] j1   -- Bessel function of 1st kind, 1st order
 *   \param[out] y0   -- Bessel function of 2nd kind, 0th order
 *   \param[out] y1   -- Bessel function of 2nd kind, 1st order
 *   \param[out] j0p  -- derivative of Bessel function of 1st kind, 0th order
 *   \param[out] j1p  -- derivative of Bessel function of 1st kind, 1st order
 *   \param[out] y0p  -- derivative of Bessel function of 2nd kind, 0th order
 *   \param[out] y1p  -- derivative of Bessel function of 2nd kind, 1st order
 *
 *   \return error code: 0 = OK, 1 = error
 */
int bessjy01a(double x, double &j0, double &j1, double &y0, double &y1, double &j0p, double &j1p, double &y0p, double &y1p);

/**
 *   Compute Bessel J and Y functions for real parameter.
 *   This algorithm computes the functions using polynomial approximations.
 *
 *   \param x    -- argument of Bessel function
 *
 *   \param[out] j0   -- Bessel function of 1st kind, 0th order
 *   \param[out] j1   -- Bessel function of 1st kind, 1st order
 *   \param[out] y0   -- Bessel function of 2nd kind, 0th order
 *   \param[out] y1   -- Bessel function of 2nd kind, 1st order
 *   \param[out] j0p  -- derivative of Bessel function of 1st kind, 0th order
 *   \param[out] j1p  -- derivative of Bessel function of 1st kind, 1st order
 *   \param[out] y0p  -- derivative of Bessel function of 2nd kind, 0th order
 *   \param[out] y1p  -- derivative of Bessel function of 2nd kind, 1st order
 *
 *   \return error code: 0 = OK, 1 = error
 */
int bessjy01b(double x, double &j0, double &j1, double &y0, double &y1, double &j0p, double &j1p, double &y0p, double &y1p);

/**
 *   Compute Bessel functions of all order up to 'n' using recurrence
 *   relations. If 'nm' < 'n' only 'nm' orders are returned.
 *
 *   \param x    -- argument of Bessel function of 1st and 2nd kind.
 *   \param n    -- order
 *
 *   \param[out] nm   -- highest order actually computed (nm <= n)
 *   \param[out] jn[] -- Bessel function of 1st kind, orders from 0 to nm
 *   \param[out] yn[] -- Bessel function of 2nd kind, orders from 0 to nm
 *   \param[out] jnp[]-- derivative of Bessel function of 1st kind, orders from 0 to nm
 *   \param[out] ynp[]-- derivative of Bessel function of 2nd kind, orders from 0 to nm
 *
 *   \return error code: 0 = OK, 1 = error
 */
int bessjyna(int n, double x,int &nm, double *jn, double *yn, double *jnp, double *ynp);

/**
 *   Compute Bessel functions of all order up to 'n' using recurrence
 *   relations. If 'nm' < 'n' only 'nm' orders are returned.
 *   Different recurrence relations used for 'x' < 300.
 *
 *   \param x    -- argument of Bessel function of 1st and 2nd kind.
 *   \param n    -- order
 *
 *   \param[out] nm   -- highest order actually computed (nm <= n)
 *   \param[out] jn[] -- Bessel function of 1st kind, orders from 0 to nm
 *   \param[out] yn[] -- Bessel function of 2nd kind, orders from 0 to nm
 *   \param[out] jnp[]-- derivative of Bessel function of 1st kind, orders from 0 to nm
 *   \param[out] ynp[]-- derivative of Bessel function of 2nd kind, orders from 0 to nm
 *
 *   \return error code: 0 = OK, 1 = error
 */
int bessjynb(int n, double x,int &nm, double *jn, double *yn, double *jnp, double *ynp);

/**
 *   The following routine computes Bessel Jv(x) and Yv(x) for
 *   arbitrary positive order (v). For negative order, use:
 *
 *       J-v(x) = Jv(x)cos(v pi) - Yv(x)sin(v pi)
 *       Y-v(x) = Jv(x)sin(v pi) + Yv(x)cos(v pi)
 *
 *   \param x    -- argument of Bessel function of 1st and 2nd kind.
 *   \param v    -- order
 *
 *   \param[out] vm   -- highest order actually computed (vm <= v)
 *   \param[out] jv[] -- Bessel function of 1st kind, orders from 0 to vm
 *   \param[out] yn[] -- Bessel function of 2nd kind, orders from 0 to vm
 *   \param[out] jvp[]-- derivative of Bessel function of 1st kind, orders from 0 to vm
 *   \param[out] yvp[]-- derivative of Bessel function of 2nd kind, orders from 0 to vm
 *
 *   \return error code: 0 = OK, 1 = error
 */
int bessjyv(double v, double x, double &vm, double *jv, double *yv, double *jvp, double *yvp);

/**
 *   Compute Bessel I and K functions for real parameter.
 *   This algorithm computes the above functions using series expansions.
 *
 *   \param x    -- argument of modified Bessel function
 *
 *   \param[out] i0   -- modified Bessel function of 1st kind, 0th order
 *   \param[out] k1   -- modified Bessel function of 1st kind, 1st order
 *   \param[out] i0   -- modified Bessel function of 2nd kind, 0th order
 *   \param[out] k1   -- modified Bessel function of 2nd kind, 1st order
 *   \param[out] i0p  -- derivative of modified Bessel function of 1st kind, 0th order
 *   \param[out] i1p  -- derivative of modified Bessel function of 1st kind, 1st order
 *   \param[out] k0p  -- derivative of modified Bessel function of 2nd kind, 0th order
 *   \param[out] k1p  -- derivative of modified Bessel function of 2nd kind, 1st order
 *
 *   \return error code: 0 = OK, 1 = error
 *
 */
int bessik01a(double x, double &i0, double &i1, double &k0, double &k1, double &i0p, double &i1p, double &k0p, double &k1p);

/**
 *   Compute Bessel I and K functions for real parameter.
 *   This algorithm computes the above functions using polynomial approximations.
 *
 *   \param x    -- argument of modified Bessel function
 *
 *   \param[out] i0   -- modified Bessel function of 1st kind, 0th order
 *   \param[out] k1   -- modified Bessel function of 1st kind, 1st order
 *   \param[out] i0   -- modified Bessel function of 2nd kind, 0th order
 *   \param[out] k1   -- modified Bessel function of 2nd kind, 1st order
 *   \param[out] i0p  -- derivative of modified Bessel function of 1st kind, 0th order
 *   \param[out] i1p  -- derivative of modified Bessel function of 1st kind, 1st order
 *   \param[out] k0p  -- derivative of modified Bessel function of 2nd kind, 0th order
 *   \param[out] k1p  -- derivative of modified Bessel function of 2nd kind, 1st order
 *
 *   \return error code: 0 = OK, 1 = error
 */
int bessik01b(double x, double &i0, double &i1, double &k0, double &k1, double &i0p, double &i1p, double &k0p, double &k1p);

/**
 *   Compute modified Bessel functions of all order up to 'n' using recurrence
 *   relations. If 'nm' < 'n' only 'nm' orders are returned.
 *
 *   \param x    -- argument of Bessel function of 1st and 2nd kind.
 *   \param n    -- order
 *
 *   \param[out] nm   -- highest order actually computed (nm <= n)
 *   \param[out] in[] -- modified Bessel function of 1st kind, orders from 0 to nm
 *   \param[out] kn[] -- modified Bessel function of 2nd kind, orders from 0 to nm
 *   \param[out] inp[]-- derivative of modified Bessel function of 1st kind, orders from 0 to nm
 *   \param[out] knp[]-- derivative of modified Bessel function of 2nd kind, orders from 0 to nm
 *
 *   \return error code: 0 = OK, 1 = error
 */
int bessikna(int n, double x,int &nm, double *in, double *kn, double *inp, double *knp);

/**
 *   Compute modified Bessel functions of all order up to 'n' using recurrence
 *   relations. If 'nm' < 'n' only 'nm' orders are returned.
 *   Different recurrence relations used for 'x' < 300.
 *
 *   \param x    -- argument of Bessel function of 1st and 2nd kind.
 *   \param n    -- order
 *
 *   \param[out] nm   -- highest order actually computed (nm <= n)
 *   \param[out] in[] -- modified Bessel function of 1st kind, orders from 0 to nm
 *   \param[out] kn[] -- modified Bessel function of 2nd kind, orders from 0 to nm
 *   \param[out] inp[]-- derivative of modified Bessel function of 1st kind, orders from 0 to nm
 *   \param[out] knp[]-- derivative of modified Bessel function of 2nd kind, orders from 0 to nm
 *
 *   \return error code: 0 = OK, 1 = error
 */
int bessiknb(int n, double x,int &nm, double *in, double *kn, double *inp, double *knp);

/**
 *   The following routine computes modified Bessel Iv(x) and Kv(x) for
 *   arbitrary positive order (v).
 *
 *   \param x    -- argument of modified Bessel function of 1st and 2nd kind.
 *   \param v    -- order
 *
 *   \param[out] vm   -- highest order actually computed (vm <= v)
 *   \param[out] iv[] -- modified Bessel function of 1st kind, orders from 0 to vm
 *   \param[out] kv[] -- modified Bessel function of 2nd kind, orders from 0 to vm
 *   \param[out] ivp[]-- derivative of modified Bessel function of 1st kind, orders from 0 to vm
 *   \param[out] kvp[]-- derivative of modified Bessel function of 2nd kind, orders from 0 to vm
 */
int bessikv(double v, double x, double &vm, double *iv, double *kv, double *ivp, double *kvp);

/**
 *   Compute Bessel J and Y functions for real parameter.
 *
 *   \param z    -- argument of Bessel function
 *
 *   \param[out] cj0   -- Bessel function of 1st kind, 0th order
 *   \param[out] cj1   -- Bessel function of 1st kind, 1st order
 *   \param[out] cy0   -- Bessel function of 2nd kind, 0th order
 *   \param[out] cy1   -- Bessel function of 2nd kind, 1st order
 *   \param[out] cj0p  -- derivative of Bessel function of 1st kind, 0th order
 *   \param[out] cj1p  -- derivative of Bessel function of 1st kind, 1st order
 *   \param[out] cy0p  -- derivative of Bessel function of 2nd kind, 0th order
 *   \param[out] cy1p  -- derivative of Bessel function of 2nd kind, 1st order
 *
 *   \return error code: 0 = OK, 1 = error
 */
int cbessjy01(std::complex<double> z, std::complex<double> &cj0, std::complex<double> &cj1,
    std::complex<double> &cy0, std::complex<double> &cy1, std::complex<double> &cj0p,
    std::complex<double> &cj1p, std::complex<double> &cy0p, std::complex<double> &cy1p);

/**
 *   Compute Bessel functions of all order up to 'n' using recurrence
 *   relations. If 'nm' < 'n' only 'nm' orders are returned.
 *
 *   \param z    -- argument of Bessel function of 1st and 2nd kind.
 *   \param n    -- order
 *
 *   \param[out] nm   -- highest order actually computed (nm <= n)
 *   \param[out] cj[] -- Bessel function of 1st kind, orders from 0 to nm
 *   \param[out] cy[] -- Bessel function of 2nd kind, orders from 0 to nm
 *   \param[out] cjp[]-- derivative of Bessel function of 1st kind, orders from 0 to nm
 *   \param[out] cyp[]-- derivative of Bessel function of 2nd kind, orders from 0 to nm
 *
 *   \return error code: 0 = OK, 1 = error
 */
int cbessjyna(int n, std::complex<double> z,int &nm, std::complex<double> *cj,
    std::complex<double> *cy, std::complex<double> *cjp, std::complex<double> *cyp);

/**
 *   Compute Bessel functions of all order up to 'n' using recurrence
 *   relations. If 'nm' < 'n' only 'nm' orders are returned.
 *   Different recurrence relations used for 'x' < 300.
 *
 *   \param z    -- argument of Bessel function of 1st and 2nd kind.
 *   \param n    -- order
 *
 *   \param[out] nm   -- highest order actually computed (nm <= n)
 *   \param[out] cj[] -- Bessel function of 1st kind, orders from 0 to nm
 *   \param[out] cy[] -- Bessel function of 2nd kind, orders from 0 to nm
 *   \param[out] cjp[]-- derivative of Bessel function of 1st kind, orders from 0 to nm
 *   \param[out] cyp[]-- derivative of Bessel function of 2nd kind, orders from 0 to nm
 *
 *   \return error code: 0 = OK, 1 = error
 */
int cbessjynb(int n, std::complex<double> z,int &nm, std::complex<double> *cj,
    std::complex<double> *cy, std::complex<double> *cjp, std::complex<double> *cyp);

/**
 *   The following routine computes Bessel Jv(x) and Yv(x) for
 *   arbitrary positive order (v). For negative order, use:
 *
 *       J-v(x) = Jv(x)cos(v pi) - Yv(x)sin(v pi)
 *       Y-v(x) = Jv(x)sin(v pi) + Yv(x)cos(v pi)
 *
 *   \param z    -- argument of Bessel function of 1st and 2nd kind.
 *   \param v    -- order
 *
 *   \param[out] vm   -- highest order actually computed (vm <= v)
 *   \param[out] cj[] -- Bessel function of 1st kind, orders from 0 to vm
 *   \param[out] cy[] -- Bessel function of 2nd kind, orders from 0 to vm
 *   \param[out] cjvp[]-- derivative of Bessel function of 1st kind, orders from 0 to vm
 *   \param[out] cyvp[]-- derivative of Bessel function of 2nd kind, orders from 0 to vm
 *
 *   \return error code: 0 = OK, 1 = error
 */
int cbessjyva(double v, std::complex<double> z, double &vm, std::complex<double>*cjv,
    std::complex<double>*cyv, std::complex<double>*cjvp, std::complex<double>*cyvp);

/**
 *   Compute Bessel I and K functions for real parameter.
 *
 *   \param z    -- argument of modified Bessel function
 *
 *   \param[out] ci0   -- modified Bessel function of 1st kind, 0th order
 *   \param[out] ck1   -- modified Bessel function of 1st kind, 1st order
 *   \param[out] ci0   -- modified Bessel function of 2nd kind, 0th order
 *   \param[out] ck1   -- modified Bessel function of 2nd kind, 1st order
 *   \param[out] ci0p  -- derivative of modified Bessel function of 1st kind, 0th order
 *   \param[out] ci1p  -- derivative of modified Bessel function of 1st kind, 1st order
 *   \param[out] ck0p  -- derivative of modified Bessel function of 2nd kind, 0th order
 *   \param[out] ck1p  -- derivative of modified Bessel function of 2nd kind, 1st order
 *
 *   \return error code: 0 = OK, 1 = error
 *
 */
int cbessik01(std::complex<double>z, std::complex<double>&ci0, std::complex<double>&ci1,
    std::complex<double>&ck0, std::complex<double>&ck1, std::complex<double>&ci0p,
    std::complex<double>&ci1p, std::complex<double>&ck0p, std::complex<double>&ck1p);

/**
 *   Compute modified Bessel functions of all order up to 'n' using recurrence
 *   relations. If 'nm' < 'n' only 'nm' orders are returned.
 *
 *   \param z    -- argument of Bessel function of 1st and 2nd kind.
 *   \param n    -- order
 *
 *   \param[out] nm   -- highest order actually computed (nm <= n)
 *   \param[out] ci[] -- modified Bessel function of 1st kind, orders from 0 to nm
 *   \param[out] ck[] -- modified Bessel function of 2nd kind, orders from 0 to nm
 *   \param[out] cip[]-- derivative of modified Bessel function of 1st kind, orders from 0 to nm
 *   \param[out] ckp[]-- derivative of modified Bessel function of 2nd kind, orders from 0 to nm
 *
 *   \return error code: 0 = OK, 1 = error
 */
int cbessikna(int n, std::complex<double> z,int &nm, std::complex<double> *ci,
    std::complex<double> *ck, std::complex<double> *cip, std::complex<double> *ckp);

/**
 *   Compute modified Bessel functions of all order up to 'n' using recurrence
 *   relations. If 'nm' < 'n' only 'nm' orders are returned.
 *   Different recurrence relations used for 'x' < 300.
 *
 *   \param z    -- argument of Bessel function of 1st and 2nd kind.
 *   \param n    -- order
 *
 *   \param[out] nm   -- highest order actually computed (nm <= n)
 *   \param[out] ci[] -- modified Bessel function of 1st kind, orders from 0 to nm
 *   \param[out] ck[] -- modified Bessel function of 2nd kind, orders from 0 to nm
 *   \param[out] cip[]-- derivative of modified Bessel function of 1st kind, orders from 0 to nm
 *   \param[out] ckp[]-- derivative of modified Bessel function of 2nd kind, orders from 0 to nm
 *
 *   \return error code: 0 = OK, 1 = error
 */
int cbessiknb(int n, std::complex<double> z,int &nm, std::complex<double> *ci,
    std::complex<double> *ck, std::complex<double> *cip, std::complex<double> *ckp);

/**
 *   The following routine computes modified Bessel Iv(x) and Kv(x) for
 *   arbitrary positive order (v).
 *
 *   \param z    -- argument of modified Bessel function of 1st and 2nd kind.
 *   \param v    -- order
 *
 *   \param[out] vm   -- highest order actually computed (vm <= v)
 *   \param[out] civ[] -- modified Bessel function of 1st kind, orders from 0 to vm
 *   \param[out] ckv[] -- modified Bessel function of 2nd kind, orders from 0 to vm
 *   \param[out] civp[]-- derivative of modified Bessel function of 1st kind, orders from 0 to vm
 *   \param[out] ckvp[]-- derivative of modified Bessel function of 2nd kind, orders from 0 to vm
 */
int cbessikv(double v, std::complex<double>z, double &vm, std::complex<double> *civ,
    std::complex<double> *ckv, std::complex<double> *civp, std::complex<double> *ckvp);


int msta1(double x,int mp);

int msta2(double x,int n,int mp);



}}}} // namespace plask::solvers::effective::bessel

#endif // PLASK__MODULE_OPTICAL_EFFECTIVE_BESSEL_HPP
