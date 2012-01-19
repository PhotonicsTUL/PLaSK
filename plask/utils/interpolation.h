#ifndef PLASK__INTERPOLATION_H
#define PLASK__INTERPOLATION_H

/** @file
This file includes interpolation utils.
*/

namespace plask { namespace interpolate {
    
    /**
     * Calculate linear interpolation (in 1d space).
     * @param p_lo, d_lo first point and value in it
     * @param p_hi, d_hi second point and value in it, must fulfill: p_lo < p_hi
     * @param p requested point
     * @return interpolate value, calculate for requested point @a p
     */
    template <typename T>
    inline T linear(double p_lo, const T& d_lo, double p_hi, const T& d_hi, double p) {
        //return ((p_hi - p) * d_lo + (p - p_lo) * d_hi) / (p_hi - p_lo);
        return d_lo + (p - p_lo) / (p_hi - p_lo) * (d_hi - d_lo);
    }
    
    /**
     * Calculate blilinear interpolation (in 2d space).
     * @param p_l, p_r, p_b, p_t left, right, bottom, top postions, must fulfill: p_l < p_r, p_b < p_t
     * @param d_lb, d_rb, d_rt, d_lt value in points: left-bottom, right-bottom, right-top, left-top
     * @param p_x, p_y requested point
     * @return interpolate value, calculate for requested point @a p_x, @a p_y
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
    
} }

#endif