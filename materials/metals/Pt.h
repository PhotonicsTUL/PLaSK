#ifndef PLASK__Pt_H
#define PLASK__Pt_H

/** @file
This file contains Pt
*/

#include <plask/material/material.h>

namespace plask { namespace materials {

/**
 * Represent Pt, its physical properties.
 */
struct Pt: public Metal {

    static constexpr const char* NAME = "Pt";

    virtual std::string name() const override;
    virtual Tensor2<double> cond(double T) const override;
    virtual Tensor2<double> thermk(double T, double t) const override;
    virtual double nr(double lam, double T, double n=0.) const override;
    virtual double absp(double lam, double T) const override;

protected:
    virtual bool isEqual(const Material& other) const override;
};


}} // namespace plask::materials

#endif	//PLASK__Pt_H
