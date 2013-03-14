#ifndef PLASK__InP_H
#define PLASK__InP_H

/** @file
This file includes undoped InP
*/

#include <plask/material/material.h>

namespace plask {

/**
 * Represent undoped InP, its physical properties.
 */
struct InP: public Semiconductor {

    static constexpr const char* NAME = "InP";

    virtual std::string name() const;
    virtual double lattC(double T, char x) const;
    virtual double Eg(double T, double eps, char point) const;
    virtual double Dso(double T, double eps) const;
    virtual Tensor2<double> Me(double T, double eps, char point) const;
    virtual Tensor2<double> Mhh(double T, double eps, char point) const;
    virtual Tensor2<double> Mlh(double T, double eps, char point) const;
    virtual double ac(double T) const;
    virtual double av(double T) const;
    virtual double b(double T) const;
    virtual double c11(double T) const;
    virtual double c12(double T) const;
    virtual Tensor2<double> thermk(double T, double t) const;
    virtual double nr(double wl, double T) const;
    virtual double absp(double wl, double T) const;

protected:
    virtual bool isEqual(const Material& other) const;

};

} // namespace plask

#endif	//PLASK__InP_H
