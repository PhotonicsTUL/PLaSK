#include "AlGaInP.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

AlGaInP::AlGaInP(const Material::Composition& Comp) {
    Al = Comp.find("Al")->second;
    Ga = Comp.find("Ga")->second;
    In = Comp.find("In")->second;
}

std::string AlGaInP::str() const { return StringBuilder("Al", Al)("Ga", Ga)("In")("P"); }

std::string AlGaInP::name() const { return NAME; }

MI_PROPERTY(AlGaInP, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlP, GaP, InP")
            )
double AlGaInP::lattC(double T, char x) const {
    return ( Al*mAlP.lattC(T,x) + Ga*mGaP.lattC(T,x) + In*mInP.lattC(T,x) );
}

MI_PROPERTY(AlGaInP, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("nonlinear interpolation: AlP, GaP, InP")
            )
double AlGaInP::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = Al*mAlP.Eg(T,e,point) + Ga*mGaP.Eg(T,e,point) + In*mInP.Eg(T,e,point)
            - Ga*In*(0.65) - Al*In*(-0.48) - Al*Ga*In*(0.18);
    else if (point == 'X') tEg = Al*mAlP.Eg(T,e,point) + Ga*mGaP.Eg(T,e,point) + In*mInP.Eg(T,e,point)
            - Al*Ga*(0.13) - Ga*In*(0.20) - Al*In*(0.38);
    else if (point == 'L') tEg = Al*mAlP.Eg(T,e,point) + Ga*mGaP.Eg(T,e,point) + In*mInP.Eg(T,e,point)
            - Ga*In*(1.03);
    else if (point == '*')
    {
        double tEgG = Al*mAlP.Eg(T,e,point) + Ga*mGaP.Eg(T,e,point) + In*mInP.Eg(T,e,point)
            - Ga*In*(0.65) - Al*In*(-0.48) - Al*Ga*In*(0.18);
        double tEgX = Al*mAlP.Eg(T,e,point) + Ga*mGaP.Eg(T,e,point) + In*mInP.Eg(T,e,point)
            - Al*Ga*(0.13) - Ga*In*(0.20) - Al*In*(0.38);
        double tEgL = Al*mAlP.Eg(T,e,point) + Ga*mGaP.Eg(T,e,point) + In*mInP.Eg(T,e,point)
            - Ga*In*(1.03);
        tEg = min(tEgG,min(tEgX,tEgL));
    }
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}

MI_PROPERTY(AlGaInP, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("nonlinear interpolation: AlP, GaP, InP"),
            MIComment("no temperature dependence")
            )
double AlGaInP::Dso(double T, double e) const {
    return ( Al*mAlP.Dso(T,e) + Ga*mGaP.Dso(T,e) + In*mInP.Dso(T,e)
             - Al*In*(-0.19) );
}

MI_PROPERTY(AlGaInP, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.230-232; "),
            MISource("nonlinear interpolation: AlP, GaP, InP"),
            MIComment("no temperature dependence")
            )
Tensor2<double> AlGaInP::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0., 0.);
    if ((point == 'G') || (point == 'X') || (point == 'L')) {
        tMe.c00 = Al*mAlP.Me(T,e,point).c00 + Ga*mGaP.Me(T,e,point).c00 + In*mInP.Me(T,e,point).c00;
        tMe.c11 = Al*mAlP.Me(T,e,point).c11 + Ga*mGaP.Me(T,e,point).c11 + In*mInP.Me(T,e,point).c11;
    }
    else if (point == '*') {
        point = 'G';
        if ( Eg(T,e,'X') == Eg(T,e,'*') ) point = 'X';
        else if ( Eg(T,e,'L') == Eg(T,e,'*') ) point = 'L';
        tMe.c00 = Al*mAlP.Me(T,e,point).c00 + Ga*mGaP.Me(T,e,point).c00 + In*mInP.Me(T,e,point).c00;
        tMe.c11 = Al*mAlP.Me(T,e,point).c11 + Ga*mGaP.Me(T,e,point).c11 + In*mInP.Me(T,e,point).c11;
    };
    if (point == 'G') {
        tMe.c00 += ( -Ga*In*(0.01854) );
        tMe.c11 += ( -Ga*In*(0.01854) );
    }
    return ( tMe );
}

MI_PROPERTY(AlGaInP, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235; "),
            MISource("linear interpolation: AlP, GaP, InP"),
            MIComment("no temperature dependence")
            )
Tensor2<double> AlGaInP::Mhh(double T, double e) const {
    double lMhh = Al*mAlP.Mhh(T,e).c00 + Ga*mGaP.Mhh(T,e).c00 + In*mInP.Mhh(T,e).c00,
           vMhh = Al*mAlP.Mhh(T,e).c11 + Ga*mGaP.Mhh(T,e).c11 + In*mInP.Mhh(T,e).c11;
    return ( Tensor2<double>(lMhh,vMhh) );
}

MI_PROPERTY(AlGaInP, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235; "),
            MISource("linear interpolation: AlP, GaP, InP"),
            MIComment("no temperature dependence")
            )
Tensor2<double> AlGaInP::Mlh(double T, double e) const {
    double lMlh = Al*mAlP.Mlh(T,e).c00 + Ga*mGaP.Mlh(T,e).c00 + In*mInP.Mlh(T,e).c00,
           vMlh = Al*mAlP.Mlh(T,e).c00 + Ga*mGaP.Mlh(T,e).c00 + In*mInP.Mlh(T,e).c00;
    return ( Tensor2<double>(lMlh,vMlh) );
}

MI_PROPERTY(AlGaInP, Mh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence; "),
            MIComment("mh = (mhh^1.5+mlh^1.5)^(2/3)")
            )
Tensor2<double> AlGaInP::Mh(double T, double e) const {
    double tMc00 = pow(pow(Mhh(T,e).c00,1.5)+pow(Mlh(T,e).c00,1.5),(2./3.));
    double tMc11 = pow(pow(Mhh(T,e).c11,1.5)+pow(Mlh(T,e).c11,1.5),(2./3.));
    Tensor2<double> tMh(tMc00, tMc11); // [001]
    return ( tMh );
}

MI_PROPERTY(AlGaInP, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double AlGaInP::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(AlGaInP, VB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlP, GaP, InP"),
            MIComment("no temperature dependence")
            )
double AlGaInP::VB(double T, double e, char point, char hole) const {
    double tVB( Al*mAlP.VB(T,0.,point,hole) + Ga*mGaP.VB(T,0.,point,hole) + In*mInP.VB(T,0.,point,hole) );
    if (!e) return tVB;
    else
    {
        double DEhy = 2.*av(T)*(1.-c12(T)/c11(T))*e;
        double DEsh = -2.*b(T)*(1.+2.*c12(T)/c11(T))*e;
        if (hole=='H') return ( tVB + DEhy - 0.5*DEsh );
        else if (hole=='L') return ( tVB + DEhy -0.5*Dso(T,e) + 0.25*DEsh + 0.5*sqrt(Dso(T,e)*Dso(T,e)+Dso(T,e)*DEsh+2.25*DEsh*DEsh) );
        else throw NotImplemented("VB can be calculated only for holes: H, L");
    }
}

MI_PROPERTY(AlGaInP, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlP, GaP, InP"),
            MIComment("no temperature dependence")
            )
double AlGaInP::ac(double T) const {
    return ( Al*mAlP.ac(T) + Ga*mGaP.ac(T) + In*mInP.ac(T) );
}

MI_PROPERTY(AlGaInP, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlP, GaP, InP"),
            MIComment("no temperature dependence")
            )
double AlGaInP::av(double T) const {
    return ( Al*mAlP.av(T) + Ga*mGaP.av(T) + In*mInP.av(T) );
}

MI_PROPERTY(AlGaInP, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlP, GaP, InP"),
            MIComment("no temperature dependence")
            )
double AlGaInP::b(double T) const {
    return ( Al*mAlP.b(T) + Ga*mGaP.b(T) + In*mInP.b(T) );
}

MI_PROPERTY(AlGaInP, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlP, GaP, InP"),
            MIComment("no temperature dependence")
            )
double AlGaInP::d(double T) const {
    return ( Al*mAlP.d(T) + Ga*mGaP.d(T) + In*mInP.d(T) );
}

MI_PROPERTY(AlGaInP, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlP, GaP, InP"),
            MIComment("no temperature dependence")
            )
double AlGaInP::c11(double T) const {
    return ( Al*mAlP.c11(T) + Ga*mGaP.c11(T) + In*mInP.c11(T) );
}

MI_PROPERTY(AlGaInP, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlP, GaP, InP"),
            MIComment("no temperature dependence")
            )
double AlGaInP::c12(double T) const {
    return ( Al*mAlP.c12(T) + Ga*mGaP.c12(T) + In*mInP.c12(T) );
}

MI_PROPERTY(AlGaInP, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlP, GaP, InP"),
            MIComment("no temperature dependence")
            )
double AlGaInP::c44(double T) const {
    return ( Al*mAlP.c44(T) + Ga*mGaP.c44(T) + In*mInP.c44(T) );
}

MI_PROPERTY(AlGaInP, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.67; "), // 300 K
            MISource("S. Adachi, Properties of Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2005) p.37; "), // temperature dependence for binaries
            MISource("inversion of nonlinear interpolation of resistivity: AlP, GaP, InP")
            )
Tensor2<double> AlGaInP::thermk(double T, double t) const {
    double lCondT = 1./(Al/mAlP.thermk(T,t).c00 + Ga/mGaP.thermk(T,t).c00 + In/mInP.thermk(T,t).c00
                        + Al*Ga*0.32 + Al*In*0.15 + Ga*In*0.72),
           vCondT = 1./(Al/mAlP.thermk(T,t).c11 + Ga/mGaP.thermk(T,t).c11 + In/mInP.thermk(T,t).c11
                        + Al*Ga*0.32 + Al*In*0.15 + Ga*In*0.72);
    return ( Tensor2<double>(lCondT,vCondT) );
}

MI_PROPERTY(AlGaInP, dens,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.18; "),
            MISource("linear interpolation: AlP, GaP, InP"),
            MIComment("no temperature dependence")
            )
double AlGaInP::dens(double T) const {
    return ( Al*mAlP.dens(T) + Ga*mGaP.dens(T) + In*mInP.dens(T) );
}

MI_PROPERTY(AlGaInP, cp,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.52; "),
            MISource("linear interpolation: AlP, GaP, InP"),
            MIComment("no temperature dependence")
            )
double AlGaInP::cp(double T) const {
    return ( Al*mAlP.cp(T) + Ga*mGaP.cp(T) + In*mInP.cp(T) );
}

Material::ConductivityType AlGaInP::condtype() const { return Material::CONDUCTIVITY_I; }

MI_PROPERTY(AlGaInP, nr,
            MIComment("TODO")
            )
double AlGaInP::nr(double wl, double T, double n) const {
    throw NotImplemented("nr for AlGaInP");
}

MI_PROPERTY(AlGaInP, absp,
            MIComment("TODO")
            )
double AlGaInP::absp(double wl, double T) const {
    throw NotImplemented("absp for AlGaInP");
}

bool AlGaInP::isEqual(const Material &other) const {
    const AlGaInP& o = static_cast<const AlGaInP&>(other);
    return o.Al == this->Al;
}

static MaterialsDB::Register<AlGaInP> materialDB_register_AlGaInP;

}} // namespace plask::materials
