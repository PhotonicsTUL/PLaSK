#include "GaInAsSb.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

GaInAsSb::GaInAsSb(const Material::Composition& Comp) {
    Ga = Comp.find("Ga")->second;
    In = Comp.find("In")->second;
    As = Comp.find("As")->second;
    Sb = Comp.find("Sb")->second;
}

std::string GaInAsSb::str() const { return StringBuilder("Ga")("In", In)("As")("Sb", Sb); }

std::string GaInAsSb::name() const { return NAME; }

MI_PROPERTY(GaInAsSb, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaSb, InSb, GaAs, InAs")
            )
double GaInAsSb::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = Ga*As*mGaAs.lattC(T,'a') + Ga*Sb*mGaSb.lattC(T,'a')
            + In*As*mInAs.lattC(T,'a') + In*Sb*mInSb.lattC(T,'a');
    else if (x == 'c') tLattC = Ga*As*mGaAs.lattC(T,'c') + Ga*Sb*mGaSb.lattC(T,'c')
            + In*As*mInAs.lattC(T,'c') + In*Sb*mInSb.lattC(T,'c');
    return ( tLattC );
}

MI_PROPERTY(GaInAsSb, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("nonlinear interpolation: GaSb, InSb, GaAs, InAs")
            )
double GaInAsSb::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = Ga*As*mGaAs.Eg(T,e,point) + Ga*Sb*mGaSb.Eg(T,e,point)
            + In*As*mInAs.Eg(T,e,point) + In*Sb*mInSb.Eg(T,e,point)
            - Ga*In*As*(0.477) - Ga*In*Sb*(0.415) - Ga*As*Sb*(1.43) - In*As*Sb*(0.67) - Ga*In*As*Sb*(0.75);
    else if (point == 'X') tEg = Ga*As*mGaAs.Eg(T,e,point) + Ga*Sb*mGaSb.Eg(T,e,point)
            + In*As*mInAs.Eg(T,e,point) + In*Sb*mInSb.Eg(T,e,point)
            - Ga*In*As*(1.4) - Ga*In*Sb*(0.33) - Ga*As*Sb*(1.2) - In*As*Sb*(0.6);
    else if (point == 'L') tEg = Ga*As*mGaAs.Eg(T,e,point) + Ga*Sb*mGaSb.Eg(T,e,point)
            + In*As*mInAs.Eg(T,e,point) + In*Sb*mInSb.Eg(T,e,point)
            - Ga*In*As*(0.33) - Ga*In*Sb*(0.4) - Ga*As*Sb*(1.2) - In*As*Sb*(0.6);
    else if (point == '*') {
        double tEgG = Ga*As*mGaAs.Eg(T,e,'G') + Ga*Sb*mGaSb.Eg(T,e,'G')
                + In*As*mInAs.Eg(T,e,'G') + In*Sb*mInSb.Eg(T,e,'G')
                - Ga*In*As*(0.477) - Ga*In*Sb*(0.415) - Ga*As*Sb*(1.43) - In*As*Sb*(0.67) - Ga*In*As*Sb*(0.75);
        double tEgX = Ga*As*mGaAs.Eg(T,e,'X') + Ga*Sb*mGaSb.Eg(T,e,'X')
                + In*As*mInAs.Eg(T,e,'X') + In*Sb*mInSb.Eg(T,e,'X')
                - Ga*In*As*(1.4) - Ga*In*Sb*(0.33) - Ga*As*Sb*(1.2) - In*As*Sb*(0.6);
        double tEgL = Ga*As*mGaAs.Eg(T,e,'L') + Ga*Sb*mGaSb.Eg(T,e,'L')
                + In*As*mInAs.Eg(T,e,'L') + In*Sb*mInSb.Eg(T,e,'L')
                - Ga*In*As*(0.33) - Ga*In*Sb*(0.4) - Ga*As*Sb*(1.2) - In*As*Sb*(0.6);
        tEg = min(tEgG,min(tEgX,tEgL));
    }
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}

MI_PROPERTY(GaInAsSb, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("nonlinear interpolation: GaSb, InSb, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double GaInAsSb::Dso(double T, double e) const {
    return ( Ga*As*mGaAs.Dso(T,e) + Ga*Sb*mGaSb.Dso(T,e)
             + In*As*mInAs.Dso(T,e) + In*Sb*mInSb.Dso(T,e)
             - Ga*In*As*(0.15) - Ga*In*Sb*(0.1) - Ga*As*Sb*(0.6) - In*As*Sb*(1.2) );
}

MI_PROPERTY(GaInAsSb, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.230-232; "),
            MISource("nonlinear interpolation: GaSb, InSb, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaInAsSb::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0., 0.);
    if ((point == 'G') || (point == 'X') || (point == 'L')) {
        tMe.c00 = Ga*As*mGaAs.Me(T,e,point).c00 + Ga*Sb*mGaSb.Me(T,e,point).c00 + In*As*mInAs.Me(T,e,point).c00 + In*Sb*mInSb.Me(T,e,point).c00;
        tMe.c11 = Ga*As*mGaAs.Me(T,e,point).c11 + Ga*Sb*mGaSb.Me(T,e,point).c11 + In*As*mInAs.Me(T,e,point).c11 + In*Sb*mInSb.Me(T,e,point).c11;
    }
    else if (point == '*') {
        point = 'G';
        if ( Eg(T,e,'X') == Eg(T,e,'*') ) point = 'X';
        else if ( Eg(T,e,'L') == Eg(T,e,'*') ) point = 'L';
        tMe.c00 = Ga*As*mGaAs.Me(T,e,point).c00 + Ga*Sb*mGaSb.Me(T,e,point).c00 + In*As*mInAs.Me(T,e,point).c00 + In*Sb*mInSb.Me(T,e,point).c00;
        tMe.c11 = Ga*As*mGaAs.Me(T,e,point).c11 + Ga*Sb*mGaSb.Me(T,e,point).c11 + In*As*mInAs.Me(T,e,point).c11 + In*Sb*mInSb.Me(T,e,point).c11;
    };
    if (point == 'G') {
        tMe.c00 += ( -Ga*In*As*(0.008)-Ga*In*Sb*(0.010)-Ga*As*Sb*(0.014)-In*As*Sb*(0.027) );
        tMe.c11 += ( -Ga*In*As*(0.008)-Ga*In*Sb*(0.010)-Ga*As*Sb*(0.014)-In*As*Sb*(0.027) );
    }
    return ( tMe );
}

MI_PROPERTY(GaInAsSb, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235; "),
            MISource("linear interpolation: GaSb, InSb, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaInAsSb::Mhh(double T, double e) const {
    double lMhh = Ga*As*mGaAs.Mhh(T,e).c00 + Ga*Sb*mGaSb.Mhh(T,e).c00
            + In*As*mInAs.Mhh(T,e).c00 + In*Sb*mInSb.Mhh(T,e).c00,
           vMhh = Ga*As*mGaAs.Mhh(T,e).c11 + Ga*Sb*mGaSb.Mhh(T,e).c11
            + In*As*mInAs.Mhh(T,e).c11 + In*Sb*mInSb.Mhh(T,e).c11;
    return ( Tensor2<double>(lMhh,vMhh) );
}

MI_PROPERTY(GaInAsSb, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235; "),
            MISource("nonlinear interpolation: GaSb, InSb, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaInAsSb::Mlh(double T, double e) const {
    double lMlh = Ga*As*mGaAs.Mlh(T,e).c00 + Ga*Sb*mGaSb.Mlh(T,e).c00
            + In*As*mInAs.Mlh(T,e).c00 + In*Sb*mInSb.Mlh(T,e).c00 - Ga*In*Sb*(0.015),
           vMlh = Ga*As*mGaAs.Mlh(T,e).c11 + Ga*Sb*mGaSb.Mlh(T,e).c11
            + In*As*mInAs.Mlh(T,e).c11 + In*Sb*mInSb.Mlh(T,e).c11 - Ga*In*Sb*(0.015);
    return ( Tensor2<double>(lMlh,vMlh) );
}

MI_PROPERTY(GaInAsSb, Mh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence; "),
            MIComment("mh = (mhh^1.5+mlh^1.5)^(2/3)")
            )
Tensor2<double> GaInAsSb::Mh(double T, double e) const {
    double tMc00 = pow(pow(Mhh(T,e).c00,1.5)+pow(Mlh(T,e).c00,1.5),(2./3.));
    double tMc11 = pow(pow(Mhh(T,e).c11,1.5)+pow(Mlh(T,e).c11,1.5),(2./3.));
    Tensor2<double> tMh(tMc00, tMc11); // [001]
    return ( tMh );
}

MI_PROPERTY(GaInAsSb, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double GaInAsSb::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(GaInAsSb, VB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("nonlinear interpolation: GaSb, InSb, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double GaInAsSb::VB(double T, double e, char point, char hole) const {
    double tVB( Ga*As*mGaAs.VB(T,0.,point,hole) + Ga*Sb*mGaSb.VB(T,0.,point,hole)
                + In*As*mInAs.VB(T,0.,point,hole) + In*Sb*mInSb.VB(T,0.,point,hole)
                - Ga*In*As*(-0.38) - Ga*As*Sb*(-1.06) );
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

MI_PROPERTY(GaInAsSb, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("nonlinear interpolation: GaSb, InSb, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double GaInAsSb::ac(double T) const {
    return ( Ga*As*mGaAs.ac(T) + Ga*Sb*mGaSb.ac(T)
             + In*As*mInAs.ac(T) + In*Sb*mInSb.ac(T) - Ga*In*As*(2.61) );
}

MI_PROPERTY(GaInAsSb, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaSb, InSb, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double GaInAsSb::av(double T) const {
    return ( Ga*As*mGaAs.av(T) + Ga*Sb*mGaSb.av(T) + In*As*mInAs.av(T) + In*Sb*mInSb.av(T) );
}

MI_PROPERTY(GaInAsSb, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaSb, InSb, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double GaInAsSb::b(double T) const {
    return ( Ga*As*mGaAs.b(T) + Ga*Sb*mGaSb.b(T) + In*As*mInAs.b(T) + In*Sb*mInSb.b(T) );
}

MI_PROPERTY(GaInAsSb, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaSb, InSb, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double GaInAsSb::d(double T) const {
    return ( Ga*As*mGaAs.d(T) + Ga*Sb*mGaSb.d(T) + In*As*mInAs.d(T) + In*Sb*mInSb.d(T) );
}

MI_PROPERTY(GaInAsSb, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaSb, InSb, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double GaInAsSb::c11(double T) const {
    return ( Ga*As*mGaAs.c11(T) + Ga*Sb*mGaSb.c11(T) + In*As*mInAs.c11(T) + In*Sb*mInSb.c11(T) );
}

MI_PROPERTY(GaInAsSb, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaSb, InSb, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double GaInAsSb::c12(double T) const {
    return ( Ga*As*mGaAs.c12(T) + Ga*Sb*mGaSb.c12(T) + In*As*mInAs.c12(T) + In*Sb*mInSb.c12(T) );
}

MI_PROPERTY(GaInAsSb, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaSb, InSb, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double GaInAsSb::c44(double T) const {
    return ( Ga*As*mGaAs.c44(T) + Ga*Sb*mGaSb.c44(T) + In*As*mInAs.c44(T) + In*Sb*mInSb.c44(T) );
}

MI_PROPERTY(GaInAsSb, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.67; "), // 300 K
            MISource("S. Adachi, Properties of Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2005) p.37; "), // temperature dependence for binaries
            MISource("inversion of nonlinear interpolation of resistivity: GaSb, InSb, GaAs, InAs")
            )
Tensor2<double> GaInAsSb::thermk(double T, double t) const {
    double lCondT = 1./(Ga*As/mGaAs.thermk(T,t).c00 + Ga*Sb/mGaSb.thermk(T,t).c00
                        + In*As/mInAs.thermk(T,t).c00 + In*Sb/mInSb.thermk(T,t).c00
                        + Ga*In*0.72 + As*Sb*0.91),
           vCondT = 1./(Ga*As/mGaAs.thermk(T,t).c11 + Ga*Sb/mGaSb.thermk(T,t).c11
                        + In*As/mInAs.thermk(T,t).c11 + In*Sb/mInSb.thermk(T,t).c11
                        + Ga*In*0.72 + As*Sb*0.91);
    return ( Tensor2<double>(lCondT,vCondT) );
}

MI_PROPERTY(GaInAsSb, dens,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.18; "),
            MISource("linear interpolation: GaSb, InSb, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double GaInAsSb::dens(double T) const {
    return ( Ga*As*mGaAs.dens(T) + Ga*Sb*mGaSb.dens(T) + In*As*mInAs.dens(T) + In*Sb*mInSb.dens(T) );
}

MI_PROPERTY(GaInAsSb, cp,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.52; "),
            MISource("linear interpolation: GaSb, InSb, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double GaInAsSb::cp(double T) const {
    return ( Ga*As*mGaAs.cp(T) + Ga*Sb*mGaSb.cp(T) + In*As*mInAs.cp(T) + In*Sb*mInSb.cp(T) );
}

Material::ConductivityType GaInAsSb::condtype() const { return Material::CONDUCTIVITY_I; }

MI_PROPERTY(GaInAsSb, nr,
            MISource("T.S. Moss, Phys. Stat. Sol. B 131 (1985) 415-427"),
            MIComment("for strained and unstrained GaInAsSb on GaSb, nr(Eg) is calculated from: nr^4 * Eg = 95 meV")
            )
double GaInAsSb::nr(double /*lam*/, double T, double /*n*/) const {
    double a0 = mGaSb.lattC(T,'a'); // typical substrate when GaInAsSb QW
    double e = a0/lattC(T,'a') - 1.;
    return pow(95./Eg(T,e,'G'),0.25);
    //return ( 3.3668 * pow(Eg(T,e,'G'),-0.32234) ); // Kumar (2010)
}

MI_PROPERTY(GaInAsSb, absp,
            MIComment("TODO")
            )
double GaInAsSb::absp(double /*lam*/, double /*T*/) const {
    throw NotImplemented("absp for GaInAsSb");
}

bool GaInAsSb::isEqual(const Material &other) const {
    const GaInAsSb& o = static_cast<const GaInAsSb&>(other);
    return o.In == this->In;
}

static MaterialsDB::Register<GaInAsSb> materialDB_register_GaInAsSb;

}} // namespace plask::materials
