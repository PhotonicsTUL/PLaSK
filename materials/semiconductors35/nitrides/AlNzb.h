#ifndef PLASK__AlNzb_H
#define PLASK__AlNzb_H

/** @file
This file contains undoped zinc-blende AlN
*/

#include <plask/material/material.h>

namespace plask { namespace materials {

/**
 * Represent undoped zinc-blende AlN, its physical properties.
 */
struct AlNzb: public Semiconductor {

    static constexpr const char* NAME = "AlNzb";

    virtual std::string name() const;

    virtual double lattC(double T, char x) const override;
    virtual double Eg(double T, double e, char point) const override;
    virtual double Dso(double T, double e) const override;
    virtual Tensor2<double> Me(double T, double e, char point) const override;
    virtual Tensor2<double> Mhh(double T, double e) const override;
    virtual Tensor2<double> Mlh(double T, double e) const override;
    virtual Tensor2<double> Mh(double T, double e) const override;
    virtual double CB(double T, double e, char point) const override;
    virtual double VB(double T, double e, char point, char hole) const override;
    virtual double ac(double T) const override;
    virtual double av(double T) const override;
    virtual double b(double T) const override;
    virtual double d(double T) const override;
    virtual double c11(double T) const override;
    virtual double c12(double T) const override;
    virtual double c44(double T) const override;
    virtual double dens(double T) const override;

  protected:

    virtual bool isEqual(const Material &other) const override;};

}} // namespace plask::materials

#endif	//PLASK__AlNzb_H
