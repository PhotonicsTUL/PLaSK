#include "InSb.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string InSb::name() const { return NAME; }

MI_PROPERTY(InSb, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double InSb::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = 6.4794 + 3.48e-5 * (T-300.);
    return ( tLattC );
}

MI_PROPERTY(InSb, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("temperature dependences for points X and L assumed to be the same as for Gamma point")
            )
double InSb::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = phys::Varshni(0.235, 0.32e-3, 170., T);
    else if (point == 'X') tEg = phys::Varshni(0.63, 0.32e-3, 170., T);
    else if (point == 'L') tEg = phys::Varshni(0.93, 0.32e-3, 170., T);
    else if (point == '*') {
        double tEgG = phys::Varshni(0.235, 0.32e-3, 170., T);
        double tEgX = phys::Varshni(0.63, 0.32e-3, 170., T);
        double tEgL = phys::Varshni(0.93, 0.32e-3, 170., T);
        tEg = min(tEgG,min(tEgX,tEgL));
    }
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}

MI_PROPERTY(InSb, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double InSb::Dso(double T, double e) const {
    return ( 0.81 );
}

MI_PROPERTY(InSb, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.230-232"),
            MIComment("no temperature dependence; "),
            MIComment("only for Gamma point")
            )
Tensor2<double> InSb::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0., 0.);
    double tMeG(0.013);
    if (point == 'G' || point == '*') {
        tMe.c00 = tMeG; tMe.c11 = tMeG;
    }
    return ( tMe );
}

MI_PROPERTY(InSb, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence")
            )
Tensor2<double> InSb::Mhh(double T, double e) const {
    Tensor2<double> tMhh(0.24, 0.24); // [001]
    return ( tMhh );
}

MI_PROPERTY(InSb, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence")
            )
Tensor2<double> InSb::Mlh(double T, double e) const {
    Tensor2<double> tMlh(0.015, 0.015); // [001]
    return ( tMlh );
}

MI_PROPERTY(InSb, Mh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence; "),
            MIComment("mh = (mhh^1.5+mlh^1.5)^(2/3)")
            )
Tensor2<double> InSb::Mh(double T, double e) const {
    double tMc00 = pow(pow(Mhh(T,e).c00,1.5)+pow(Mlh(T,e).c00,1.5),(2./3.));
    double tMc11 = pow(pow(Mhh(T,e).c11,1.5)+pow(Mlh(T,e).c11,1.5),(2./3.));
    Tensor2<double> tMh(tMc00, tMc11); // [001]
    return ( tMh );
}

MI_PROPERTY(InSb, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double InSb::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point) + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(InSb, VB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double InSb::VB(double T, double e, char point, char hole) const {
    double tVB(0.);
    if (e) {
        double DEhy = 2.*av(T)*(1.-c12(T)/c11(T))*e;
        double DEsh = -2.*b(T)*(1.+2.*c12(T)/c11(T))*e;
        if (hole=='H') return ( tVB + DEhy - 0.5*DEsh );
        else if (hole=='L') return ( tVB + DEhy -0.5*Dso(T,e) + 0.25*DEsh + 0.5*sqrt(Dso(T,e)*Dso(T,e)+Dso(T,e)*DEsh+2.25*DEsh*DEsh) );
        else throw NotImplemented("VB can be calculated only for holes: H, L");
    }
    return tVB;
}

MI_PROPERTY(InSb, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double InSb::ac(double T) const {
    return ( -6.94 );
}

MI_PROPERTY(InSb, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double InSb::av(double T) const {
    return ( 0.36 );
}

MI_PROPERTY(InSb, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double InSb::b(double T) const {
    return ( -2.0 );
}

MI_PROPERTY(InSb, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double InSb::d(double T) const {
    return ( -4.7 );
}

MI_PROPERTY(InSb, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double InSb::c11(double T) const {
    return ( 68.47 );
}

MI_PROPERTY(InSb, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double InSb::c12(double T) const {
    return ( 37.35 );
}

MI_PROPERTY(InSb, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double InSb::c44(double T) const {
    return ( 31.11 );
}

MI_PROPERTY(InSb, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.67; "), // 300 K
            MISource("S. Adachi, Properties of Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2005) p.37"), // temperature dependence
            MIArgumentRange(MaterialInfo::T, 20, 300)
           )
Tensor2<double> InSb::thermk(double T, double t) const {
    double tCondT = (1./0.0571)*pow((300./T),1.60);
    return ( Tensor2<double>(tCondT, tCondT) );
}

MI_PROPERTY(InSb, dens,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.18"),
            MIComment("no temperature dependence")
            )
double InSb::dens(double T) const { return 5.77677e3; }

MI_PROPERTY(InSb, cp,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.52"),
            MIComment("no temperature dependence")
            )
double InSb::cp(double T) const { return 0.350e3; }

Material::ConductivityType InSb::condtype() const { return Material::CONDUCTIVITY_I; }

bool InSb::isEqual(const Material &other) const {
    return true;
}

static MaterialsDB::Register<InSb> materialDB_register_InSb;

}} // namespace plask::materials
