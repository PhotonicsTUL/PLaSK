#ifndef PLASK__AlxOy_H
#define PLASK__AlxOy_H

/** @file
This file contains AlxOy
*/

#include <plask/material/material.h>

namespace plask { namespace materials {

/**
 * Represent AlxOy, its physical properties.
 */
struct AlxOy: public Dielectric {

    static constexpr const char* NAME = "AlxOy";

    virtual std::string name() const override;
    virtual Tensor2<double> cond(double T) const override;
    virtual Tensor2<double> thermk(double T, double h=INFINITY) const override;
    virtual double nr(double wl, double T, double n = .0) const override;
    virtual double absp(double wl, double T) const override;

protected:
    virtual bool isEqual(const Material& other) const override;

};


}} // namespace plask::materials

#endif	//PLASK__AlxOy_H
