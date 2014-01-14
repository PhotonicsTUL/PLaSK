#ifndef PLASK__AuGe_H
#define PLASK__AuGe_H

/** @file
This file contains AuGe
*/

#include <plask/material/material.h>

namespace plask { namespace materials {

/**
 * Represent AuGe, its physical properties.
 */
struct AuGe: public Metal {

    static constexpr const char* NAME = "AuGe";

    virtual std::string name() const override;
    virtual Tensor2<double> cond(double T) const override;
    virtual Tensor2<double> thermk(double T, double t) const override;
    virtual double nr(double wl, double T, double n=0.) const override;
    virtual double absp(double wl, double T) const override;
protected:
    virtual bool isEqual(const Material& other) const override;
};


}} // namespace plask::materials

#endif	//PLASK__AuGe_H
