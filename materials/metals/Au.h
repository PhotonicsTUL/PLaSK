#ifndef PLASK__Au_H
#define PLASK__Au_H

/** @file
This file includes Au
*/

#include <plask/material/material.h>

namespace plask {

/**
 * Represent Au, its physical properties.
 */
struct Au: public Metal {

    static constexpr const char* NAME = "Au";

    virtual std::string name() const;
    virtual Tensor2<double> cond(double T) const;
    virtual Tensor2<double> thermk(double T, double t) const;
    virtual double nr(double wl, double T) const;
    virtual double absp(double wl, double T) const;
protected:
    virtual bool isEqual(const Material& other) const;
};


} // namespace plask

#endif	//PLASK__Au_H
