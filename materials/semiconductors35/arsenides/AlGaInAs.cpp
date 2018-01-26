#include "AlGaInAs.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

AlGaInAs::AlGaInAs(const Material::Composition& Comp) {
    Al = Comp.find("Al")->second;
    Ga = Comp.find("Ga")->second;
    In = Comp.find("In")->second;
}

std::string AlGaInAs::str() const { return StringBuilder("Al", Al)("Ga", Ga)("In")("As"); }

std::string AlGaInAs::name() const { return NAME; }

MI_PROPERTY(AlGaInAs, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlAs, GaAs, InAs")
            )
double AlGaInAs::lattC(double T, char x) const {
    return ( Al*mAlAs.lattC(T,x) + Ga*mGaAs.lattC(T,x) + In*mInAs.lattC(T,x) );
}

MI_PROPERTY(AlGaInAs, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("nonlinear interpolation: AlAs, GaAs, InAs")
            )
double AlGaInAs::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = Al*mAlAs.Eg(T,e,point) + Ga*mGaAs.Eg(T,e,point) + In*mInAs.Eg(T,e,point)
            - Al*Ga*(-0.127+1.310*Al) - Ga*In*(0.477) - Al*In*(0.70) - Al*Ga*In*(0.22);
    else if (point == 'X') tEg = Al*mAlAs.Eg(T,e,point) + Ga*mGaAs.Eg(T,e,point) + In*mInAs.Eg(T,e,point)
            - Al*Ga*(0.055) - Ga*In*(1.4);
    else if (point == 'L') tEg = Al*mAlAs.Eg(T,e,point) + Ga*mGaAs.Eg(T,e,point) + In*mInAs.Eg(T,e,point)
            - Ga*In*(0.33);
    else if (point == '*') {
        double tEgG = Al*mAlAs.Eg(T,e,'G') + Ga*mGaAs.Eg(T,e,'G') + In*mInAs.Eg(T,e,'G')
                - Al*Ga*(-0.127+1.310*Al) - Ga*In*(0.477) - Al*In*(0.70) - Al*Ga*In*(0.22);
        double tEgX = Al*mAlAs.Eg(T,e,'X') + Ga*mGaAs.Eg(T,e,'X') + In*mInAs.Eg(T,e,'X')
                - Al*Ga*(0.055) - Ga*In*(1.4);
        double tEgL = Al*mAlAs.Eg(T,e,'L') + Ga*mGaAs.Eg(T,e,'L') + In*mInAs.Eg(T,e,'L')
                - Ga*In*(0.33);
        tEg = min(tEgG,min(tEgX,tEgL));
    }
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}

MI_PROPERTY(AlGaInAs, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("nonlinear interpolation: AlAs, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double AlGaInAs::Dso(double T, double e) const {
    return ( Al*mAlAs.Dso(T,e) + Ga*mGaAs.Dso(T,e) + In*mInAs.Dso(T,e)
             - Ga*In*(0.15) - Al*In*(0.15) );
}

MI_PROPERTY(AlGaInAs, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.230-232; "),
            MISource("nonlinear interpolation: AlAs, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
Tensor2<double> AlGaInAs::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0., 0.);
    if ((point == 'G') || (point == 'X') || (point == 'L')) {
        tMe.c00 = Al*mAlAs.Me(T,e,point).c00 + Ga*mGaAs.Me(T,e,point).c00 + In*mInAs.Me(T,e,point).c00;
        tMe.c11 = Al*mAlAs.Me(T,e,point).c11 + Ga*mGaAs.Me(T,e,point).c11 + In*mInAs.Me(T,e,point).c11;
    }
    else if (point == '*') {
        point = 'G';
        if ( Eg(T,e,'X') == Eg(T,e,'*') ) point = 'X';
        else if ( Eg(T,e,'L') == Eg(T,e,'*') ) point = 'L';
        tMe.c00 = Al*mAlAs.Me(T,e,point).c00 + Ga*mGaAs.Me(T,e,point).c00 + In*mInAs.Me(T,e,point).c00;
        tMe.c11 = Al*mAlAs.Me(T,e,point).c11 + Ga*mGaAs.Me(T,e,point).c11 + In*mInAs.Me(T,e,point).c11;
    };
    if (point == 'G') {
        tMe.c00 += ( -Ga*In*(0.008)-Al*In*(0.012) );
        tMe.c11 += ( -Ga*In*(0.008)-Al*In*(0.012) );
    }
    return ( tMe );
}

MI_PROPERTY(AlGaInAs, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235; "),
            MISource("linear interpolation: AlAs, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
Tensor2<double> AlGaInAs::Mhh(double T, double e) const {
    double lMhh = Al*mAlAs.Mhh(T,e).c00 + Ga*mGaAs.Mhh(T,e).c00 + In*mInAs.Mhh(T,e).c00,
           vMhh = Al*mAlAs.Mhh(T,e).c11 + Ga*mGaAs.Mhh(T,e).c11 + In*mInAs.Mhh(T,e).c11;
    return ( Tensor2<double>(lMhh,vMhh) );
}

MI_PROPERTY(AlGaInAs, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235; "),
            MISource("linear interpolation: AlAs, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
Tensor2<double> AlGaInAs::Mlh(double T, double e) const {
    double lMlh = Al*mAlAs.Mlh(T,e).c00 + Ga*mGaAs.Mlh(T,e).c00 + In*mInAs.Mlh(T,e).c00,
           vMlh = Al*mAlAs.Mlh(T,e).c00 + Ga*mGaAs.Mlh(T,e).c00 + In*mInAs.Mlh(T,e).c00;
    return ( Tensor2<double>(lMlh,vMlh) );
}

MI_PROPERTY(AlGaInAs, Mh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence; "),
            MIComment("mh = (mhh^1.5+mlh^1.5)^(2/3)")
            )
Tensor2<double> AlGaInAs::Mh(double T, double e) const {
    double tMc00 = pow(pow(Mhh(T,e).c00,1.5)+pow(Mlh(T,e).c00,1.5),(2./3.));
    double tMc11 = pow(pow(Mhh(T,e).c11,1.5)+pow(Mlh(T,e).c11,1.5),(2./3.));
    Tensor2<double> tMh(tMc00, tMc11); // [001]
    return ( tMh );
}

MI_PROPERTY(AlGaInAs, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double AlGaInAs::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(AlGaInAs, VB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("nonlinear interpolation: AlAs, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double AlGaInAs::VB(double T, double e, char point, char hole) const {
    double tVB( Al*mAlAs.VB(T,0.,point,hole) + Ga*mGaAs.VB(T,0.,point,hole) + In*mInAs.VB(T,0.,point,hole)
                - Ga*In*(-0.38) );
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

MI_PROPERTY(AlGaInAs, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("nonlinear interpolation: AlAs, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double AlGaInAs::ac(double T) const {
    return ( Al*mAlAs.ac(T) + Ga*mGaAs.ac(T) + In*mInAs.ac(T)
             - Ga*In*(2.61) );
}

MI_PROPERTY(AlGaInAs, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlAs, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double AlGaInAs::av(double T) const {
    return ( Al*mAlAs.av(T) + Ga*mGaAs.av(T) + In*mInAs.av(T) );
}

MI_PROPERTY(AlGaInAs, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlAs, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double AlGaInAs::b(double T) const {
    return ( Al*mAlAs.b(T) + Ga*mGaAs.b(T) + In*mInAs.b(T) );
}

MI_PROPERTY(AlGaInAs, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlAs, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double AlGaInAs::d(double T) const {
    return ( Al*mAlAs.d(T) + Ga*mGaAs.d(T) + In*mInAs.d(T) );
}

MI_PROPERTY(AlGaInAs, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlAs, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double AlGaInAs::c11(double T) const {
    return ( Al*mAlAs.c11(T) + Ga*mGaAs.c11(T) + In*mInAs.c11(T) );
}

MI_PROPERTY(AlGaInAs, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlAs, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double AlGaInAs::c12(double T) const {
    return ( Al*mAlAs.c12(T) + Ga*mGaAs.c12(T) + In*mInAs.c12(T) );
}

MI_PROPERTY(AlGaInAs, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlAs, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double AlGaInAs::c44(double T) const {
    return ( Al*mAlAs.c44(T) + Ga*mGaAs.c44(T) + In*mInAs.c44(T) );
}

MI_PROPERTY(AlGaInAs, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.67; "), // 300 K
            MISource("S. Adachi, Properties of Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2005) p.37; "), // temperature dependence for binaries
            MISource("inversion of nonlinear interpolation of resistivity: AlAs, GaAs, InAs")
            )
Tensor2<double> AlGaInAs::thermk(double T, double t) const {
    double lCondT = 1./(Al/mAlAs.thermk(T,t).c00 + Ga/mGaAs.thermk(T,t).c00 + In/mInAs.thermk(T,t).c00
                        + Al*Ga*0.32 + Al*In*0.15 + Ga*In*0.72),
           vCondT = 1./(Al/mAlAs.thermk(T,t).c11 + Ga/mGaAs.thermk(T,t).c11 + In/mInAs.thermk(T,t).c11
                        + Al*Ga*0.32 + Al*In*0.15 + Ga*In*0.72);
    return ( Tensor2<double>(lCondT,vCondT) );
}

MI_PROPERTY(AlGaInAs, dens,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.18; "),
            MISource("linear interpolation: AlAs, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double AlGaInAs::dens(double T) const {
    return ( Al*mAlAs.dens(T) + Ga*mGaAs.dens(T) + In*mInAs.dens(T) );
}

MI_PROPERTY(AlGaInAs, cp,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.52; "),
            MISource("linear interpolation: AlAs, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double AlGaInAs::cp(double T) const {
    return ( Al*mAlAs.cp(T) + Ga*mGaAs.cp(T) + In*mInAs.cp(T) );
}

Material::ConductivityType AlGaInAs::condtype() const { return Material::CONDUCTIVITY_I; }

MI_PROPERTY(AlGaInAs, nr,
            MIComment("TODO")
            )
double AlGaInAs::nr(double /*lam*/, double /*T*/, double /*n*/) const {
    throw NotImplemented("nr for AlGaInAs");
}

MI_PROPERTY(AlGaInAs, absp,
            MIComment("TODO")
            )
double AlGaInAs::absp(double /*lam*/, double /*T*/) const {
    throw NotImplemented("absp for AlGaInAs");
}

bool AlGaInAs::isEqual(const Material &other) const {
    const AlGaInAs& o = static_cast<const AlGaInAs&>(other);
    return o.Al == this->Al;
}

static MaterialsDB::Register<AlGaInAs> materialDB_register_AlGaInAs;

}} // namespace plask::materials
