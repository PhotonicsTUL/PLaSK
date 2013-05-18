#ifndef PLASK__AlxOy_H
#define PLASK__AlxOy_H

/** @file
This file contains AlxOy
*/

#include <plask/material/material.h>

namespace plask {

/**
 * Represent AlxOy, its physical properties.
 */
struct AlxOy: public Dielectric {

    static constexpr const char* NAME = "AlxOy";

    virtual std::string name() const;
    virtual Tensor2<double> cond(double T) const;
    virtual Tensor2<double> thermk(double T, double h=INFINITY) const;
    virtual double nr(double wl, double T) const;
    virtual double absp(double wl, double T) const;

protected:
    virtual bool isEqual(const Material& other) const;

};


} // namespace plask

#endif	//PLASK__AlxOy_H
