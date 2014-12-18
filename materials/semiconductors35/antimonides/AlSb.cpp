#include "AlSb.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string AlSb::name() const { return NAME; }

MI_PROPERTY(AlSb, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double AlSb::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = 6.1355 + 2.60e-5 * (T-300.);
    return ( tLattC );
}

MI_PROPERTY(AlSb, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double AlSb::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = phys::Varshni(2.386, 0.42e-3, 140., T);
    else if (point == 'X') tEg = phys::Varshni(1.696, 0.39e-3, 140., T);
    else if (point == 'L') tEg = phys::Varshni(2.329, 0.58e-3, 140., T);
    else if (point == '*')
    {
        double tEgG = phys::Varshni(2.386, 0.42e-3, 140., T);
        double tEgX = phys::Varshni(1.696, 0.39e-3, 140., T);
        double tEgL = phys::Varshni(2.329, 0.58e-3, 140., T);
        tEg = min(tEgG,min(tEgX,tEgL));
    }
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}

MI_PROPERTY(AlSb, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double AlSb::Dso(double T, double e) const {
    return ( 0.676 );
}

MI_PROPERTY(AlSb, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.230-232"),
            MIComment("no temperature dependence")
            )
Tensor2<double> AlSb::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0., 0.);
    double tMeG(0.14), tMeX(0.84), tMeL(1.05);
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

MI_PROPERTY(AlSb, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence")
            )
Tensor2<double> AlSb::Mhh(double T, double e) const {
    Tensor2<double> tMhh(0.47, 0.47); // [001]
    return ( tMhh );
}

MI_PROPERTY(AlSb, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence")
            )
Tensor2<double> AlSb::Mlh(double T, double e) const {
    Tensor2<double> tMlh(0.16, 0.16); // [001]
    return ( tMlh );
}

MI_PROPERTY(AlSb, Mh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence; "),
            MIComment("mh = (mhh^1.5+mlh^1.5)^(2/3)")
            )
Tensor2<double> AlSb::Mh(double T, double e) const {
    double tMc00 = pow(pow(Mhh(T,e).c00,1.5)+pow(Mlh(T,e).c00,1.5),(2./3.));
    double tMc11 = pow(pow(Mhh(T,e).c11,1.5)+pow(Mlh(T,e).c11,1.5),(2./3.));
    Tensor2<double> tMh(tMc00, tMc11); // [001]
    return ( tMh );
}

MI_PROPERTY(AlSb, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double AlSb::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point) + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(AlSb, VB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double AlSb::VB(double T, double e, char point, char hole) const {
    double tVB(-0.41);
    if (e) {
        double DEhy = 2.*av(T)*(1.-c12(T)/c11(T))*e;
        double DEsh = -2.*b(T)*(1.+2.*c12(T)/c11(T))*e;
        if (hole=='H') return ( tVB + DEhy - 0.5*DEsh );
        else if (hole=='L') return ( tVB + DEhy -0.5*Dso(T,e) + 0.25*DEsh + 0.5*sqrt(Dso(T,e)*Dso(T,e)+Dso(T,e)*DEsh+2.25*DEsh*DEsh) );
        else throw NotImplemented("VB can be calculated only for holes: H, L");
    }
    return tVB;
}

MI_PROPERTY(AlSb, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double AlSb::ac(double T) const {
    return ( -4.5 );
}

MI_PROPERTY(AlSb, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double AlSb::av(double T) const {
    return ( 1.4 );
}

MI_PROPERTY(AlSb, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double AlSb::b(double T) const {
    return ( -1.35 );
}

MI_PROPERTY(AlSb, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double AlSb::d(double T) const {
    return ( -4.3 );
}

MI_PROPERTY(AlSb, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double AlSb::c11(double T) const {
    return ( 87.69 );
}

MI_PROPERTY(AlSb, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double AlSb::c12(double T) const {
    return ( 43.41 );
}

MI_PROPERTY(AlSb, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double AlSb::c44(double T) const {
    return ( 40.76 );
}

MI_PROPERTY(AlSb, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.67; "), // 300 K
            MISource("S. Adachi, Properties of Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2005) p.37"), // temperature dependence
            MIArgumentRange(MaterialInfo::T, 300, 945)
           )
Tensor2<double> AlSb::thermk(double T, double t) const {
    double tCondT = (1./0.0175)*pow((300./T),1.42);
    return ( Tensor2<double>(tCondT, tCondT) );
}

MI_PROPERTY(AlSb, dens,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.18"),
            MIComment("no temperature dependence")
            )
double AlSb::dens(double T) const { return 4.2775e3; }

MI_PROPERTY(AlSb, cp,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.52"),
            MIComment("no temperature dependence")
            )
double AlSb::cp(double T) const { return 0.326e3; }

Material::ConductivityType AlSb::condtype() const { return Material::CONDUCTIVITY_I; }

bool AlSb::isEqual(const Material &other) const {
    return true;
}

static MaterialsDB::Register<AlSb> materialDB_register_AlSb;

}} // namespace plask::materials
