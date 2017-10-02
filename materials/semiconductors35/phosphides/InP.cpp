#include "InP.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string InP::name() const { return NAME; }

MI_PROPERTY(InP, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double InP::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = 5.8697 + 2.79e-5 * (T-300.);
    return ( tLattC );
}

MI_PROPERTY(InP, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double InP::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = phys::Varshni(1.4236, 0.363e-3, 162., T);
    else if (point == 'X') tEg = 2.384-3.7e-4*T;
    else if (point == 'L') tEg = phys::Varshni(2.014, 0.363e-3, 162., T);
    else if (point == '*') {
        double tEgG = phys::Varshni(1.4236, 0.363e-3, 162., T);
        double tEgX = 2.384-3.7e-4*T;
        double tEgL = phys::Varshni(2.014, 0.363e-3, 162., T);
        tEg = min(tEgG,min(tEgX,tEgL));
    }
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}

MI_PROPERTY(InP, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double InP::Dso(double T, double e) const {
    return ( 0.108 );
}

MI_PROPERTY(InP, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.230-232"),
            MIComment("no temperature dependence")
            )
Tensor2<double> InP::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0., 0.);
    double tMeG(0.07927), tMeX(1.09), tMeL(0.76);
    if (point == 'G') {
        tMe.c00 = tMeG; tMe.c11 = tMeG;
    }
    else if (point == 'X') {
        tMe.c00 = tMeX; tMe.c11 = tMeX;
    }
    else if (point == 'L') {
        tMe.c00 = tMeL; tMe.c11 = tMeL;
    }
    else if (point == '*') {
        if ( Eg(T,e,'G') == Eg(T,e,'*') ) {
            tMe.c00 = tMeG; tMe.c11 = tMeG;
        }
        else if ( Eg(T,e,'X') == Eg(T,e,'*') ) {
            tMe.c00 = tMeX; tMe.c11 = tMeX;
        }
        else if ( Eg(T,e,'L') == Eg(T,e,'*') ) {
            tMe.c00 = tMeL; tMe.c11 = tMeL;
        }
    }
    return ( tMe );
}

MI_PROPERTY(InP, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence")
            )
Tensor2<double> InP::Mhh(double T, double e) const {
    Tensor2<double> tMhh(0.46, 0.46); // [001]
    return ( tMhh );
}

MI_PROPERTY(InP, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence")
            )
Tensor2<double> InP::Mlh(double T, double e) const {
    Tensor2<double> tMlh(0.12, 0.12);
    return ( tMlh );
}

MI_PROPERTY(InP, Mh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence; "),
            MIComment("mh = (mhh^1.5+mlh^1.5)^(2/3)")
            )
Tensor2<double> InP::Mh(double T, double e) const {
    double tMc00 = pow(pow(Mhh(T,e).c00,1.5)+pow(Mlh(T,e).c00,1.5),(2./3.));
    double tMc11 = pow(pow(Mhh(T,e).c11,1.5)+pow(Mlh(T,e).c11,1.5),(2./3.));
    Tensor2<double> tMh(tMc00, tMc11); // [001]
    return ( tMh );
}

MI_PROPERTY(InP, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double InP::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(InP, VB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double InP::VB(double T, double e, char point, char hole) const {
    double tVB(-0.94);
    if (e) {
        double DEhy = 2.*av(T)*(1.-c12(T)/c11(T))*e;
        double DEsh = -2.*b(T)*(1.+2.*c12(T)/c11(T))*e;
        if (hole=='H') return ( tVB + DEhy - 0.5*DEsh );
        else if (hole=='L') return ( tVB + DEhy -0.5*Dso(T,e) + 0.25*DEsh + 0.5*sqrt(Dso(T,e)*Dso(T,e)+Dso(T,e)*DEsh+2.25*DEsh*DEsh) );
        else throw NotImplemented("VB can be calculated only for holes: H, L");
    }
    return tVB;
}

MI_PROPERTY(InP, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double InP::ac(double T) const {
    return ( -6.0 );
}

MI_PROPERTY(InP, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double InP::av(double T) const {
    return ( 0.6 );
}

MI_PROPERTY(InP, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double InP::b(double T) const {
    return ( -2.0 );
}

MI_PROPERTY(InP, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double InP::d(double T) const {
    return ( -5.0 );
}

MI_PROPERTY(InP, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double InP::c11(double T) const {
    return ( 101.1 );
}

MI_PROPERTY(InP, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double InP::c12(double T) const {
    return ( 56.1 );
}

MI_PROPERTY(InP, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double InP::c44(double T) const {
    return ( 45.6 );
}

MI_PROPERTY(InP, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.67; "), // 300 K
            MISource("S. Adachi, Properties of Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2005) p.37"), // temperature dependence
            MIArgumentRange(MaterialInfo::T, 20, 800)
            )
Tensor2<double> InP::thermk(double T, double t) const {
    double tCondT = 68.*pow((300./T),1.42);
    return ( Tensor2<double>(tCondT, tCondT) );
}

MI_PROPERTY(InP, dens,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.18"),
            MIComment("no temperature dependence")
            )
double InP::dens(double T) const { return 4.7902e3; }

MI_PROPERTY(InP, cp,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.52"),
            MIComment("no temperature dependence")
            )
double InP::cp(double T) const { return 0.322e3; }

Material::ConductivityType InP::condtype() const { return Material::CONDUCTIVITY_I; }

MI_PROPERTY(InP, nr,
            MISource("refractiveindex.info, Handbook of Optics, 2nd edition, Vol. 2. McGraw-Hill 1994"),
            MISource("S. Adachi, Handbook on Physical Properties of Semiconductors, vol. 2 III-V Compound Semiconductors, Chapter 16, Kluwer Academic Publishers, 2004"),
            MIArgumentRange(MaterialInfo::lam, 950, 10000)
            )
double InP::nr(double lam, double T, double n) const {
    double  twl = lam/1e3,
            tnr = sqrt(7.255+(2.316*twl*twl)/(twl*twl-0.6263*0.6263)+2.765*twl*twl/(twl*twl-32.935*32.935)),
            tBeta = 2.7e-5; //S. Adachi (long-wavelength limit)
    return ( tnr + tBeta*(T-300.) );
}

MI_PROPERTY(InP, absp,
            MIComment("TODO")
            )
double InP::absp(double lam, double T) const {
    throw NotImplemented("absp for InP");
}

bool InP::isEqual(const Material &other) const {
    return true;
}

static MaterialsDB::Register<InP> materialDB_register_InP;

}} // namespace plask::materials
