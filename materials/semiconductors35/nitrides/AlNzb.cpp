#include "AlNzb.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

MI_PROPERTY(AlNzb, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696")
           )
double AlNzb::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = 4.38 + 2.05e-5 * (T-300.); // TODO: what instead 2.05e-5 - find in: S. Wang, Phys. Status Solidi, B Basic Res. 246, 1618 (2009)
    return tLattC;
}

MI_PROPERTY(AlNzb, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696")
           )
double AlNzb::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = phys::Varshni(5.4, 0.593e-3, 600., T);
    else if (point == 'X') tEg = phys::Varshni(4.9, 0.593e-3, 600., T);
    else if (point == 'L') tEg = phys::Varshni(9.3, 0.593e-3, 600., T);
    else if (point == '*') {
        double tEgG = phys::Varshni(5.4, 0.593e-3, 600., T);
        double tEgX = phys::Varshni(4.9, 0.593e-3, 600., T);
        double tEgL = phys::Varshni(9.3, 0.593e-3, 600., T);
        tEg = min(tEgG,min(tEgX,tEgL));
    }
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}

MI_PROPERTY(AlNzb, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MIComment("no temperature dependence")
           )
double AlNzb::Dso(double /*T*/, double /*e*/) const {
    return 0.019;
}

MI_PROPERTY(AlNzb, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.230-232"),
            MIComment("only for Gamma and X points; "),
            MIComment("no temperature dependence")
           )
Tensor2<double> AlNzb::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0., 0.);
    double tMeG(0.26), tMeX(0.78);
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

MI_PROPERTY(AlNzb, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence")
           )
Tensor2<double> AlNzb::Mhh(double /*T*/, double /*e*/) const {
    return Tensor2<double>(1.15, 1.15); // [001]
}

MI_PROPERTY(AlNzb, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence")
           )
Tensor2<double> AlNzb::Mlh(double /*T*/, double /*e*/) const {
    return Tensor2<double>(0.39, 0.39);
}

MI_PROPERTY(AlNzb, Mh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence; "),
            MIComment("mh = (mhh^1.5+mlh^1.5)^(2/3)")
            )
Tensor2<double> AlNzb::Mh(double T, double e) const {
    double tMc00 = pow(pow(Mhh(T,e).c00,1.5)+pow(Mlh(T,e).c00,1.5),(2./3.));
    double tMc11 = pow(pow(Mhh(T,e).c11,1.5)+pow(Mlh(T,e).c11,1.5),(2./3.));
    Tensor2<double> tMh(tMc00, tMc11); // [001]
    return ( tMh );
}

MI_PROPERTY(AlNzb, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696")
           )
double AlNzb::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return tCB;
    else return tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e;
}

MI_PROPERTY(AlNzb, VB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MIComment("no temperature dependence")
           )
double AlNzb::VB(double T, double e, char /*point*/, char hole) const {
    double tVB(-3.44);
    if (e) {
        double DEhy = 2.*av(T)*(1.-c12(T)/c11(T))*e;
        double DEsh = -2.*b(T)*(1.+2.*c12(T)/c11(T))*e;
        if (hole=='H') return ( tVB + DEhy - 0.5*DEsh );
        else if (hole=='L') return ( tVB + DEhy -0.5*Dso(T,e) + 0.25*DEsh + 0.5*sqrt(Dso(T,e)*Dso(T,e)+Dso(T,e)*DEsh+2.25*DEsh*DEsh) );
        else throw NotImplemented("VB can be calculated only for holes: H, L");
    }
    return tVB;
}

MI_PROPERTY(AlNzb, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MIComment("no temperature dependence")
           )
double AlNzb::ac(double /*T*/) const {
    return -4.5;
}

MI_PROPERTY(AlNzb, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MIComment("no temperature dependence")
           )
double AlNzb::av(double /*T*/) const {
    return -4.9;
}

MI_PROPERTY(AlNzb, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MIComment("no temperature dependence")
           )
double AlNzb::b(double /*T*/) const {
    return -1.7;
}

MI_PROPERTY(AlNzb, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MIComment("no temperature dependence")
           )
double AlNzb::d(double /*T*/) const {
    return -5.5;
}

MI_PROPERTY(AlNzb, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MIComment("no temperature dependence")
           )
double AlNzb::c11(double /*T*/) const {
    return 304.;
}

MI_PROPERTY(AlNzb, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MIComment("no temperature dependence")
           )
double AlNzb::c12(double /*T*/) const {
    return 160.;
}

MI_PROPERTY(AlNzb, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MIComment("no temperature dependence")
           )
double AlNzb::c44(double /*T*/) const {
    return 193.;
}

MI_PROPERTY(AlNzb, dens,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.18"),
            MIComment("no temperature dependence")
            )
double AlNzb::dens(double /*T*/) const { return 3.24e3; }

std::string AlNzb::name() const { return NAME; }

bool AlNzb::isEqual(const Material &/*other*/) const {
    return true;
}

static MaterialsDB::Register<AlNzb> materialDB_register_AlNzb;

}} // namespace plask::materials
