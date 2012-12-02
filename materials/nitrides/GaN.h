#ifndef PLASK__GaN_H
#define PLASK__GaN_H

/** @file
This file includes undoped GaN
*/

#include <plask/material/material.h>

namespace plask {

/**
 * Represent undoped GaN, its physical properties.
 */
struct GaN: public Semiconductor {

    static constexpr const char* NAME = "GaN";

    virtual std::string name() const;
    virtual Tensor2<double> cond(double T) const;
    virtual Tensor2<double> thermk(double T, double t) const;
    virtual double nr(double wl, double T) const;
    virtual double absp(double wl, double T) const;
    virtual double lattC(double T, char x) const;
    virtual double Eg(double T, char point) const;
    virtual Tensor2<double> Me(double T, char point) const;
    virtual Tensor2<double> Mhh(double T, char point) const;
    virtual Tensor2<double> Mlh(double T, char point) const;

};


} // namespace plask

#endif	//PLASK__GaN_H
