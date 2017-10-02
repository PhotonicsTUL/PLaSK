#ifndef PLASK__aSi_H
#define PLASK__aSi_H

/** @file
This file contains a-Si
*/

#include <plask/material/material.h>

namespace plask { namespace materials {

/**
 * Represent a-Si, its physical properties.
 */
struct aSi: public Dielectric {

    static constexpr const char* NAME = "Si";

    virtual std::string name() const override;
    virtual Tensor2<double> cond(double T) const override;
    virtual Tensor2<double> thermk(double T, double h=INFINITY) const override;
    virtual ConductivityType condtype() const override;
    virtual double nr(double lam, double T, double n = .0) const override;
    virtual double absp(double lam, double T) const override;

protected:
    virtual bool isEqual(const Material& other) const override;

};


}} // namespace plask::materials

#endif	//PLASK__aSi_H
