#ifndef PLASK__AlOx_H
#define PLASK__AlOx_H

/** @file
This file contains AlOx
*/

#include <plask/material/material.h>

namespace plask { namespace materials {

/**
 * Represent AlOx, its physical properties.
 */
struct AlOx: public Dielectric {

    static constexpr const char* NAME = "AlOx";

    virtual std::string name() const override;
    virtual Tensor2<double> cond(double T) const override;
    virtual Tensor2<double> thermk(double T, double h=INFINITY) const override;
    virtual double nr(double wl, double T, double n = .0) const override;
    virtual double absp(double wl, double T) const override;

protected:
    virtual bool isEqual(const Material& other) const override;

};


}} // namespace plask::materials

#endif	//PLASK__AlOx_H
