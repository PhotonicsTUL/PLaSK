#ifndef PLASK__AuGe_H
#define PLASK__AuGe_H

/** @file
This file includes AuGe
*/

#include <plask/material/material.h>

namespace plask {

/**
 * Represent AuGe, its physical properties.
 */
struct AuGe: public Metal {

    static constexpr const char* NAME = "AuGe";

    virtual std::string name() const;
    virtual Tensor2<double> cond(double T) const;
    virtual Tensor2<double> thermk(double T, double t) const;
    virtual double nr(double wl, double T) const;
    virtual double absp(double wl, double T) const;
protected:
    virtual bool isEqual(const Material& other) const;
};


} // namespace plask

#endif	//PLASK__AuGe_H
