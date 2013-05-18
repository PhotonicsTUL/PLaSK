#ifndef PLASK__Ni_H
#define PLASK__Ni_H

/** @file
This file contains Ni
*/

#include <plask/material/material.h>

namespace plask {

/**
 * Represent Ni, its physical properties.
 */
struct Ni: public Metal {

    static constexpr const char* NAME = "Ni";

    virtual std::string name() const;
    virtual Tensor2<double> cond(double T) const;
    virtual Tensor2<double> thermk(double T, double t) const;
    virtual double nr(double wl, double T) const;
    virtual double absp(double wl, double T) const;

protected:
    virtual bool isEqual(const Material& other) const;
};


} // namespace plask

#endif	//PLASK__Ni_H
