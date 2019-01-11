#ifndef PLASK__Ni_H
#define PLASK__Ni_H

/** @file
This file contains Ni
*/

#include <plask/material/material.h>

namespace plask { namespace materials {

/**
 * Represent Ni, its physical properties.
 */
struct Ni: public Metal {

    static constexpr const char* NAME = "Ni";

    virtual std::string name() const override;
    virtual Tensor2<double> cond(double T) const override;
    virtual Tensor2<double> thermk(double T, double t) const override;
	virtual double nr(double lam, double T=300., double n=0.) const override;
	virtual double absp(double lam, double T=300.) const override;

protected:
    virtual bool isEqual(const Material& other) const override;
};


}} // namespace plask::materials

#endif	//PLASK__Ni_H
