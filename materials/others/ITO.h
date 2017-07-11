#ifndef PLASK__ITO_H
#define PLASK__ITO_H

/** @file
This file contains ITO (indium tim oxide)
*/

#include <plask/material/material.h>

namespace plask { namespace materials {

/**
 * Represent ITO (indium tim oxide), its physical properties.
 */
struct ITO: public Semiconductor {

    static constexpr const char* NAME = "ITO";

    virtual std::string name() const override;
    virtual Tensor2<double> cond(double T) const override;
    virtual Tensor2<double> thermk(double T, double t) const override;
    virtual double nr(double wl, double T, double n=0.) const override;
    virtual double absp(double wl, double T) const override;
protected:
    virtual bool isEqual(const Material& other) const override;
};


}} // namespace plask::materials

#endif	//PLASK__ITO_H
