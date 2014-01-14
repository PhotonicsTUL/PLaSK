#ifndef PLASK__AlN_H
#define PLASK__AlN_H

/** @file
This file contains undoped AlN
*/

#include <plask/material/material.h>

namespace plask { namespace materials {

/**
 * Represent undoped AlN, its physical properties.
 */
struct AlN: public Semiconductor {

    static constexpr const char* NAME = "AlN";

	virtual std::string name() const override;
    virtual Tensor2<double> thermk(double T, double t) const override;
    virtual double nr(double wl, double T, double n=0.) const override;
    virtual double absp(double wl, double T) const override;
    virtual double lattC(double T, char x) const override;
    virtual double Eg(double T, double e, char point) const override;
    virtual Tensor2<double> Me(double T, double e, char point) const override;
    virtual Tensor2<double> Mhh(double T, double e) const override;
    virtual Tensor2<double> Mlh(double T, double e) const override;
/*TODO
    virtual double Mhh(double T, double e, char point) const override;
    virtual double Mhh_l(double T, char point) const override;
    virtual double Mhh_v(double T, char point) const override;
    virtual double Mlh(double T, double e, char point) const override;
    virtual double Mlh_l(double T, char point) const override;
    virtual double Mlh_v(double T, char point) const override;
*/

protected:
    virtual bool isEqual(const Material& other) const override;

};


}} // namespace plask::materials

#endif	//PLASK__AlN_H
