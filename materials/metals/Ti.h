#ifndef PLASK__Ti_H
#define PLASK__Ti_H

/** @file
This file contains Ti
*/

#include <plask/material/material.h>

namespace plask {

/**
 * Represent Ti, its physical properties.
 */
struct Ti: public Metal {

    static constexpr const char* NAME = "Ti";

    virtual std::string name() const;
    virtual Tensor2<double> cond(double T) const;
    virtual Tensor2<double> thermk(double T, double t) const;
    virtual double nr(double wl, double T) const;
    virtual double absp(double wl, double T) const;

protected:
    virtual bool isEqual(const Material& other) const;
};


} // namespace plask

#endif	//PLASK__Ti_H
