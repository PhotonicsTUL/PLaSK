#ifndef PLASK__OPTICAL_EFFECTIVE_BISECTION_H
#define PLASK__OPTICAL_EFFECTIVE_BISECTION_H

#include <plask/plask.hpp>

namespace plask { namespace solvers { namespace effective {

struct Contour {

    const Solver* solver;           ///< Solver that created this contour

    const std::function<dcomplex(dcomplex)>& fun; ///< Function being investigated

    double re0,                     ///< Real part of the lower left corner of the contour
           im0,                     ///< Real part of the lower left corner of the contour
           re1,                     ///< Imaginary part of the upper right corner of the contour
           im1;                     ///< Imaginary part of the upper right corner of the contour

    DataVector<dcomplex> bottom,    ///< Vector of computed function values at bottom side of the contour
                         right,     ///< Vector of computed function values at right side of the contour
                         top,       ///< Vector of computed function values at top side of the contour
                         left;      ///< Vector of computed function values at left side of the contour

    Contour(const Solver* solver, const std::function<dcomplex(dcomplex)>& fun): solver(solver), fun(fun) {};

    Contour(const Contour& src): solver(src.solver), fun(src.fun), re0(src.re0), im0(src.im0), re1(src.re1), im1(src.im1),
                                 bottom(src.bottom), right(src.right), top(src.top), left(src.left) {}

    Contour(Contour&& src): solver(src.solver), fun(src.fun), re0(src.re0), im0(src.im0), re1(src.re1), im1(src.im1),
                            bottom(std::move(src.bottom)), right(std::move(src.right)), top(std::move(src.top)), left(std::move(src.left)) {}

    /**
     * Create contour in specified range
     * \param solver solver that created this contour
     * \param fun function to compute
     * \param corner0,corner1 corners of the integral
     * \param ren,imn number of contour points along each real and imaginary axis, respectively
     */
    Contour(const Solver* solver, const std::function<dcomplex(dcomplex)>& fun, dcomplex corner0, dcomplex corner1, size_t ren, size_t imn);

    /**
     * Compute winding number of the contour
     * \return winding number of the contour
     */
    int winding() const {
        return   crossings(bottom, re0,im0, re1,im0)
               + crossings(right,  re1,im0, re1,im1)
               - crossings(top,    re1,im1, re0,im1)
               - crossings(left,   re0,im1, re0,im0);
    }

    /**
     * Divide contour.
     * The contour is always divided across longer (by number of points) axis.
     * In case of equal axes, the imaginary one is cut.
     * \return pair of two contours
     */
    std::pair<Contour,Contour> divide() const;

  private:

    /// Count crossings of negative real half-axis
    int crossings(const DataVector<dcomplex>& line, double r0, double i0, double r1, double i1) const;

};

/**
 * Global complex bisection algorithm
 * \param solver solver that created this contour
 * \param fun function to compute
 * \param corner0,corner1 corners of the integral
 * \param resteps,imsteps number of contour points along each real and imaginary axis, respectively
 * \param eps desired precision
 * \return list of found zeros
 */
std::vector<dcomplex> findZeros(const Solver* solver, const std::function<dcomplex(dcomplex)>& fun,
                                dcomplex corner0, dcomplex corner1, size_t resteps, size_t imsteps, dcomplex eps);

}}} // namespace plask::solvers::effective

#endif // PLASK__OPTICAL_EFFECTIVE_BISECTION_H