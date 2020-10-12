#ifndef PLASK__UTILS_INTERPOLATION_H
#define PLASK__UTILS_INTERPOLATION_H

/** @file
This file contains interpolation utils.
*/

namespace plask { namespace interpolation {

    /**
     * Calculate linear interpolation (in 1D space).
     * @param p_lo, d_lo first point and value in it
     * @param p_hi, d_hi second point and value in it, must fulfill: p_lo < p_hi
     * @param p requested point
     * @return interpolated value, calculated for requested point @a p
     */
    template <typename T>
    inline T linear(double p_lo, const T& d_lo, double p_hi, const T& d_hi, double p) {
        //return ((p_hi - p) * d_lo + (p - p_lo) * d_hi) / (p_hi - p_lo);
        return d_lo + (p - p_lo) / (p_hi - p_lo) * (d_hi - d_lo);
    }

    /**
     * Calculate blilinear interpolation (in 2D space).
     * @param p_l, p_r, p_b, p_t left, right, bottom, top postions, must fulfill: p_l < p_r, p_b < p_t
     * @param d_lb, d_rb, d_rt, d_lt value in points: left-bottom, right-bottom, right-top, left-top
     * @param p_x, p_y requested point
     * @return interpolated value, calculated for requested point @a p_x, @a p_y
     */
    template <typename T>
    inline T bilinear(double p_l, double p_r, double p_b, double p_t,
                      const T& d_lb, const T& d_rb, const T& d_rt, const T& d_lt,
                      double p_x, double p_y) {
        const double delta_x_hi = p_r - p_x;
        const double delta_x_lo = p_x - p_l;
        return ((d_lb * delta_x_hi + d_rb * delta_x_lo) * (p_t - p_y) +
                (d_lt * delta_x_hi + d_rt * delta_x_lo) * (p_y - p_b))
                / (p_t - p_b) / (p_r - p_l);
    }

    /**
     * Calculate trililinear interpolation (in 3D space).
     * @param p_l, p_r, p_b, p_t left, right, bottom, top postions, must fulfill: p_l < p_r, p_b < p_t
     * @param p_lo, p_hi positions of points in thirth direction, must fulfill p_lo < p_hi
     * @param lo_d_lb, lo_d_rb, lo_d_rt, lo_d_lt value in points: lo-left-bottom, lo-right-bottom, lo-right-top, lo-left-top
     * @param hi_d_lb, hi_d_rb, hi_d_rt, hi_d_lt value in points: hi-left-bottom, hi-right-bottom, hi-right-top, hi-left-top
     * @param p_x, p_y, p_lohi requested point
     * @return interpolated value, calculate for requested point @a p_x, @a p_y, @a p_lohi
     */
    template <typename T>
    inline T trilinear(double p_l, double p_r, double p_b, double p_t, double p_lo, double p_hi,
                      const T& lo_d_lb, const T& lo_d_rb, const T& lo_d_rt, const T& lo_d_lt,
                      const T& hi_d_lb, const T& hi_d_rb, const T& hi_d_rt, const T& hi_d_lt,
                      double p_x, double p_y, double p_lohi) {
        return linear(p_lo, bilinear(p_l, p_r, p_b, p_t, lo_d_lb, lo_d_rb, lo_d_rt, lo_d_lt, p_x, p_y),
                      p_hi, bilinear(p_l, p_r, p_b, p_t, hi_d_lb, hi_d_rb, hi_d_rt, hi_d_lt, p_x, p_y),
                      p_lohi);
    }

} }

#endif // PLASK__UTILS_INTERPOLATION_H
