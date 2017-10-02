#ifndef PLASK__In_H
#define PLASK__In_H

/** @file
This file contains In
*/

#include <plask/material/material.h>

namespace plask { namespace materials {

/**
 * Represent In, its physical properties.
 */
struct In: public Metal {

    static constexpr const char* NAME = "In";

    virtual std::string name() const override;
    virtual Tensor2<double> cond(double T) const override;
    virtual Tensor2<double> thermk(double T, double t) const override;
    virtual double nr(double lam, double T, double n=0.) const override;
    virtual double absp(double lam, double T) const override;

protected:
    virtual bool isEqual(const Material& other) const override;
};


}} // namespace plask::materials

#endif	//PLASK__In_H
