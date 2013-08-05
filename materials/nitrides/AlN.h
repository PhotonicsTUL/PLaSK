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

	virtual std::string name() const;
    virtual Tensor2<double> thermk(double T, double t) const;
    virtual double nr(double wl, double T) const;
    virtual double absp(double wl, double T) const;
    virtual double lattC(double T, char x) const;
    virtual double Eg(double T, double e, char point) const;
    virtual Tensor2<double> Me(double T, double e, char point) const;
    virtual Tensor2<double> Mhh(double T, double e) const;
    virtual Tensor2<double> Mlh(double T, double e) const;
/*TODO
    virtual double Mhh(double T, double e, char point) const;
    virtual double Mhh_l(double T, char point) const;
    virtual double Mhh_v(double T, char point) const;
    virtual double Mlh(double T, double e, char point) const;
    virtual double Mlh_l(double T, char point) const;
    virtual double Mlh_v(double T, char point) const;
*/

protected:
    virtual bool isEqual(const Material& other) const;

};


}} // namespace plask::materials

#endif	//PLASK__AlN_H
