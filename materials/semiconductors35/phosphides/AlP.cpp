#include "AlP.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string AlP::name() const { return NAME; }

MI_PROPERTY(AlP, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double AlP::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = 5.4672 + 2.92e-5 * (T-300.);
    return ( tLattC );
}

MI_PROPERTY(AlP, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double AlP::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = phys::Varshni(3.63, 0.5771e-3, 372., T);
    else if (point == 'X') tEg = phys::Varshni(2.52, 0.318e-3, 588., T);
    else if (point == 'L') tEg = phys::Varshni(3.57, 0.318e-3, 588., T);
    else if (point == '*') {
        double tEgG = phys::Varshni(3.63, 0.5771e-3, 372., T);
        double tEgX = phys::Varshni(2.52, 0.318e-3, 588., T);
        double tEgL = phys::Varshni(3.57, 0.318e-3, 588., T);
        tEg = min(tEgG,min(tEgX,tEgL));
    }
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}

MI_PROPERTY(AlP, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double AlP::Dso(double /*T*/, double /*e*/) const {
    return ( 0.07 );
}

MI_PROPERTY(AlP, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.230-232"),
            MIComment("only for Gamma and X points; "),
            MIComment("no temperature dependence")
            )
Tensor2<double> AlP::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0., 0.);
    double tMeG(0.220), tMeX(1.14);
    if (point == 'G') {
        tMe.c00 = tMeG; tMe.c11 = tMeG;
    }
    else if (point == 'X') {
        tMe.c00 = tMeX; tMe.c11 = tMeX;
    }
    else if (point == '*') {
        if ( Eg(T,e,'G') == Eg(T,e,'*') ) {
            tMe.c00 = tMeG; tMe.c11 = tMeG;
        }
        else if ( Eg(T,e,'X') == Eg(T,e,'*') ) {
            tMe.c00 = tMeX; tMe.c11 = tMeX;
        }
    }
    return ( tMe );
}

MI_PROPERTY(AlP, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence")
            )
Tensor2<double> AlP::Mhh(double /*T*/, double /*e*/) const {
    return Tensor2<double>(0.30, 0.30); // [001];
}

MI_PROPERTY(AlP, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence")
            )
Tensor2<double> AlP::Mlh(double /*T*/, double /*e*/) const {
    return Tensor2<double>(0.28, 0.28);
}

MI_PROPERTY(AlP, Mh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence; "),
            MIComment("mh = (mhh^1.5+mlh^1.5)^(2/3)")
            )
Tensor2<double> AlP::Mh(double T, double e) const {
    double tMc00 = pow(pow(Mhh(T,e).c00,1.5)+pow(Mlh(T,e).c00,1.5),(2./3.));
    double tMc11 = pow(pow(Mhh(T,e).c11,1.5)+pow(Mlh(T,e).c11,1.5),(2./3.));
    Tensor2<double> tMh(tMc00, tMc11); // [001]
    return ( tMh );
}

MI_PROPERTY(AlP, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double AlP::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point) + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(AlP, VB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double AlP::VB(double T, double e, char /*point*/, char hole) const {
    double tVB(-1.74);
    if (e) {
        double DEhy = 2.*av(T)*(1.-c12(T)/c11(T))*e;
        double DEsh = -2.*b(T)*(1.+2.*c12(T)/c11(T))*e;
        if (hole=='H') return ( tVB + DEhy - 0.5*DEsh );
        else if (hole=='L') return ( tVB + DEhy -0.5*Dso(T,e) + 0.25*DEsh + 0.5*sqrt(Dso(T,e)*Dso(T,e)+Dso(T,e)*DEsh+2.25*DEsh*DEsh) );
        else throw NotImplemented("VB can be calculated only for holes: H, L");
    }
    return tVB;
}

MI_PROPERTY(AlP, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double AlP::ac(double /*T*/) const {
    return ( -5.7 );
}

MI_PROPERTY(AlP, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double AlP::av(double /*T*/) const {
    return ( 3.0 );
}

MI_PROPERTY(AlP, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double AlP::b(double /*T*/) const {
    return ( -1.5 );
}

MI_PROPERTY(AlP, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double AlP::d(double /*T*/) const {
    return ( -4.6 );
}

MI_PROPERTY(AlP, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double AlP::c11(double /*T*/) const {
    return ( 133.0 );
}

MI_PROPERTY(AlP, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double AlP::c12(double /*T*/) const {
    return ( 63.0 );
}

MI_PROPERTY(AlP, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double AlP::c44(double /*T*/) const {
    return ( 61.5 );
}

MI_PROPERTY(AlP, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.67; "), // 300 K
            MISource("S. Adachi, Properties of Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2005) p.37"), // temperature dependence
            MIComment("temperature dependence assumed to be the same as for AlSb")
            )
Tensor2<double> AlP::thermk(double T, double /*t*/) const {
    double tCondT = (1./0.011)*pow((300./T),1.42);
    return ( Tensor2<double>(tCondT, tCondT) );
}

MI_PROPERTY(AlP, dens,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.18"),
            MIComment("no temperature dependence")
            )
double AlP::dens(double /*T*/) const { return 2.3604e3; }

MI_PROPERTY(AlP, cp,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.52"),
            MIComment("no temperature dependence")
            )
double AlP::cp(double /*T*/) const { return 0.727e3; }

Material::ConductivityType AlP::condtype() const { return Material::CONDUCTIVITY_I; }

MI_PROPERTY(AlP, nr,
            MIComment("TODO")
            )
double AlP::nr(double /*lam*/, double /*T*/, double /*n*/) const {
    throw NotImplemented("nr for AlP");
}

MI_PROPERTY(AlP, absp,
            MIComment("TODO")
            )
double AlP::absp(double /*lam*/, double /*T*/) const {
    throw NotImplemented("absp for AlP");
}

bool AlP::isEqual(const Material &/*other*/) const {
    return true;
}

static MaterialsDB::Register<AlP> materialDB_register_AlP;

}} // namespace plask::materials
