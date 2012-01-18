#ifndef PLASK__INTERPOLATION_H
#define PLASK__INTERPOLATION_H

/** @file
This file includes interpolation utils.
*/

namespace plask { namespace interpolate {
    
    /**
     * Linear interpolate in 1d space.
     * @param p1, d1 first point and value in it
     * @param p2, d2 second point and value in it
     * @param p requested point, must fulfill: p1 <= p <= p2
     * @return interpolate value, calculate for requested point @a p
     */
    template <typename T>
    inline T linear(double p1, const T& d1, double p2, const T& d2, double p) {
        //TODO which one is more stable?
        //auto d0 = data[index-1];
        //return d0 + (data[index] - d0) * (point - operator[](index-1)) / (operator[](index) - operator[](index-1));
        return ((p2 - p) * d1 + (p - p1) * d2) / (p2 - p1);
    }
    
} }

#endif