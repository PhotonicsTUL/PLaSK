#ifndef PLASK__Pt_H
#define PLASK__Pt_H

/** @file
This file includes Pt
*/

#include <plask/material/material.h>

namespace plask {

/**
 * Represent Pt, its physical properties.
 */
struct Pt: public Metal {

    static constexpr const char* NAME = "Pt";

    virtual std::string name() const;
    virtual Tensor2<double> cond(double T) const;
    virtual Tensor2<double> thermk(double T, double t) const;
    virtual double nr(double wl, double T) const;
    virtual double absp(double wl, double T) const;

protected:
    virtual bool isEqual(const Material& other) const;
};


} // namespace plask

#endif	//PLASK__Pt_H
