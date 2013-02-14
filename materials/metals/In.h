#ifndef PLASK__In_H
#define PLASK__In_H

/** @file
This file includes In
*/

#include <plask/material/material.h>

namespace plask {

/**
 * Represent In, its physical properties.
 */
struct In: public Metal {

    static constexpr const char* NAME = "In";

    virtual std::string name() const;
    virtual Tensor2<double> cond(double T) const;
    virtual Tensor2<double> thermk(double T, double t) const;
    virtual double nr(double wl, double T) const;
    virtual double absp(double wl, double T) const;

protected:
    virtual bool isEqual(const Material& other) const;
};


} // namespace plask

#endif	//PLASK__In_H
