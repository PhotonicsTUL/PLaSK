#ifndef PLASK__AlAs_H
#define PLASK__AlAs_H

/** @file
This file includes undoped AlAs
*/

#include <plask/material/material.h>

namespace plask {

/**
 * Represent undoped AlAs, its physical properties.
 */
struct AlAs: public Semiconductor {

    static constexpr const char* NAME = "AlAs";

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
protected:
    virtual bool isEqual(const Material& other) const;
};

} // namespace plask

#endif	//PLASK__AlAs_H
