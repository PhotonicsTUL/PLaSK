#ifndef PLASK__InAs_H
#define PLASK__InAs_H

/** @file
This file contains undoped InAs
*/

#include <plask/material/material.h>

namespace plask { namespace materials {

/**
 * Represent undoped InAs, its physical properties.
 */
struct InAs: public Semiconductor {

    static constexpr const char* NAME = "InAs";

    virtual std::string name() const { return NAME; }

    virtual double lattC(double T, char x) const {
        double tLattC(0.);
        if (x == 'a') tLattC = 6.0583 + 2.74e-5 * (T-300.);
        return tLattC;
    }

    virtual double Eg(double T, double e, char point) const {
        double tEg(0.);
        if (point == 'G') tEg = phys::Varshni(0.417, 0.276e-3, 93., T);
        else if (point == 'X') tEg = phys::Varshni(1.433, 0.276e-3, 93., T);
        else if (point == 'L') tEg = phys::Varshni(1.133, 0.276e-3, 93., T);
        return tEg;
    }

    virtual double Dso(double T, double e) const {
        return 0.39;
    }

    virtual Tensor2<double> Me(double T, double e, char point) const {
        Tensor2<double> tMe(0., 0.);
        if (point == 'G') {
            tMe.c00 = 0.024;
            tMe.c11 = 0.024;
        }
        return tMe;
    }

    virtual Tensor2<double> Mhh(double T, double e) const {
        Tensor2<double> tMhh(0.26, 0.26); // [001]
        return tMhh;
    }

    virtual Tensor2<double> Mlh(double T, double e) const {
        Tensor2<double> tMlh(0.027, 0.027);
        return tMlh;
    }

    virtual double CB(double T, double e, char point) const {
        double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
        if (!e) return tCB;
        else return tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e;
    }

    virtual double VB(double T, double e, char point, char hole) const {
        double tVB(-0.59);
        if (!e) return tVB;
        else
        {
            double DEhy = 2.*av(T)*(1.-c12(T)/c11(T))*e;
            double DEsh = -2.*b(T)*(1.+2.*c12(T)/c11(T))*e;
            if (hole=='H') return ( tVB + DEhy - 0.5*DEsh );
            else if (hole=='L') return ( tVB + DEhy -0.5*Dso(T,e) + 0.25*DEsh + 0.5*sqrt(Dso(T,e)*Dso(T,e)+Dso(T,e)*DEsh+2.25*DEsh*DEsh) );
        }
    }

    virtual double ac(double T) const {
        return -5.08;
    }

    virtual double av(double T) const {
        return 1.00;
    }

    virtual double b(double T) const {
        return -1.8;
    }

    virtual double d(double T) const {
        return -3.6;
    }

    virtual double c11(double T) const {
        return 83.29;
    }

    virtual double c12(double T) const {
        return 45.26;
    }

    virtual double c44(double T) const {
        return 39.59;
    }

    virtual Tensor2<double> thermk(double T, double t) const {
        double tCondT = 30.*pow((300./T),1.234);
        return(Tensor2<double>(tCondT, tCondT));
    }

  protected:
      
    virtual bool isEqual(const Material &other) const {
        return true;
    }

};


}} // namespace plask::materials

#endif	//PLASK__InAs_H
