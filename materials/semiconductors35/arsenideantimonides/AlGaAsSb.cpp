#include "AlGaAsSb.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

AlGaAsSb::AlGaAsSb(const Material::Composition& Comp) {
    Al = Comp.find("Al")->second;
    Ga = Comp.find("Ga")->second;
    As = Comp.find("As")->second;
    Sb = Comp.find("Sb")->second;
}

std::string AlGaAsSb::str() const { return StringBuilder("Al", Al)("Ga")("As")("Sb", Sb); }

std::string AlGaAsSb::name() const { return NAME; }

MI_PROPERTY(AlGaAsSb, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaSb, AlSb, GaAs, AlAs")
            )
double AlGaAsSb::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = Ga*As*mGaAs.lattC(T,'a') + Ga*Sb*mGaSb.lattC(T,'a')
            + Al*As*mAlAs.lattC(T,'a') + Al*Sb*mAlSb.lattC(T,'a');
    else if (x == 'c') tLattC = Ga*As*mGaAs.lattC(T,'c') + Ga*Sb*mGaSb.lattC(T,'c')
            + Al*As*mAlAs.lattC(T,'c') + Al*Sb*mAlSb.lattC(T,'c');
    return ( tLattC );
}

MI_PROPERTY(AlGaAsSb, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("nonlinear interpolation: GaSb, AlSb, GaAs, AlAs")
            )
double AlGaAsSb::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = Ga*As*mGaAs.Eg(T,e,point) + Ga*Sb*mGaSb.Eg(T,e,point)
            + Al*As*mAlAs.Eg(T,e,point) + Al*Sb*mAlSb.Eg(T,e,point)
            - Al*Ga*As*(-0.127+1.310*Al) - Al*Ga*Sb*(-0.044+1.22*Al) - Ga*As*Sb*(1.43) - Al*As*Sb*(0.8) - Al*Ga*As*Sb*0.48;
    else if (point == 'X') tEg = Ga*As*mGaAs.Eg(T,e,point) + Ga*Sb*mGaSb.Eg(T,e,point)
            + Al*As*mAlAs.Eg(T,e,point) + Al*Sb*mAlSb.Eg(T,e,point)
            - Al*Ga*As*(0.055) - Ga*As*Sb*(1.2) - Al*As*Sb*(0.28);
    else if (point == 'L') tEg = Ga*As*mGaAs.Eg(T,e,point) + Ga*Sb*mGaSb.Eg(T,e,point)
            + Al*As*mAlAs.Eg(T,e,point) + Al*Sb*mAlSb.Eg(T,e,point)
            - Ga*As*Sb*(1.2) - Al*As*Sb*(0.28);
    else if (point == '*')
    {
        double tEgG = Ga*As*mGaAs.Eg(T,e,'G') + Ga*Sb*mGaSb.Eg(T,e,'G')
                + Al*As*mAlAs.Eg(T,e,'G') + Al*Sb*mAlSb.Eg(T,e,'G')
                - Al*Ga*As*(-0.127+1.310*Al) - Al*Ga*Sb*(-0.044+1.22*Al) - Ga*As*Sb*(1.43) - Al*As*Sb*(0.8) - Al*Ga*As*Sb*0.48;
        double tEgX = Ga*As*mGaAs.Eg(T,e,'X') + Ga*Sb*mGaSb.Eg(T,e,'X')
                + Al*As*mAlAs.Eg(T,e,'X') + Al*Sb*mAlSb.Eg(T,e,'X')
                - Al*Ga*As*(0.055) - Ga*As*Sb*(1.2) - Al*As*Sb*(0.28);
        double tEgL = Ga*As*mGaAs.Eg(T,e,'L') + Ga*Sb*mGaSb.Eg(T,e,'L')
                + Al*As*mAlAs.Eg(T,e,'L') + Al*Sb*mAlSb.Eg(T,e,'L')
                - Ga*As*Sb*(1.2) - Al*As*Sb*(0.28);
        tEg = min(tEgG,min(tEgX,tEgL));
    }
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}

MI_PROPERTY(AlGaAsSb, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("nonlinear interpolation: GaSb, AlSb, GaAs, AlAs"),
            MIComment("no temperature dependence")
            )
double AlGaAsSb::Dso(double T, double e) const {
    return ( Ga*As*mGaAs.Dso(T,e) + Ga*Sb*mGaSb.Dso(T,e)
             + Al*As*mAlAs.Dso(T,e) + Al*Sb*mAlSb.Dso(T,e)
             - Al*Ga*Sb*(0.3) - Ga*As*Sb*(0.6) - Al*As*Sb*(0.15) );
}

MI_PROPERTY(AlGaAsSb, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.230-232; "),
            MISource("nonlinear interpolation: GaSb, AlSb, GaAs, AlAs"),
            MIComment("no temperature dependence")
            )
Tensor2<double> AlGaAsSb::Me(double T, double e, char point) const {
    double lMe = Ga*As*mGaAs.Me(T,e,point).c00 + Ga*Sb*mGaSb.Me(T,e,point).c00
            + Al*As*mAlAs.Me(T,e,point).c00 + Al*Sb*mAlSb.Me(T,e,point).c00
            - Ga*As*Sb*(0.014),
           vMe = Ga*As*mGaAs.Me(T,e,point).c11 + Ga*Sb*mGaSb.Me(T,e,point).c11
            + Al*As*mAlAs.Me(T,e,point).c11 + Al*Sb*mAlSb.Me(T,e,point).c11
            - Ga*As*Sb*(0.014);
    return ( Tensor2<double>(lMe,vMe) );
}

MI_PROPERTY(AlGaAsSb, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235; "),
            MISource("linear interpolation: GaSb, AlSb, GaAs, AlAs"),
            MIComment("no temperature dependence")
            )
Tensor2<double> AlGaAsSb::Mhh(double T, double e) const {
    double lMhh = Ga*As*mGaAs.Mhh(T,e).c00 + Ga*Sb*mGaSb.Mhh(T,e).c00
            + Al*As*mAlAs.Mhh(T,e).c00 + Al*Sb*mAlSb.Mhh(T,e).c00,
           vMhh = Ga*As*mGaAs.Mhh(T,e).c11 + Ga*Sb*mGaSb.Mhh(T,e).c11
            + Al*As*mAlAs.Mhh(T,e).c11 + Al*Sb*mAlSb.Mhh(T,e).c11;
    return ( Tensor2<double>(lMhh,vMhh) );
}

MI_PROPERTY(AlGaAsSb, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235; "),
            MISource("nonlinear interpolation: GaSb, AlSb, GaAs, AlAs"),
            MIComment("no temperature dependence")
            )
Tensor2<double> AlGaAsSb::Mlh(double T, double e) const {
    double lMlh = Ga*As*mGaAs.Mlh(T,e).c00 + Ga*Sb*mGaSb.Mlh(T,e).c00
            + Al*As*mAlAs.Mlh(T,e).c00 + Al*Sb*mAlSb.Mlh(T,e).c00,
           vMlh = Ga*As*mGaAs.Mlh(T,e).c11 + Ga*Sb*mGaSb.Mlh(T,e).c11
            + Al*As*mAlAs.Mlh(T,e).c11 + Al*Sb*mAlSb.Mlh(T,e).c11;
    return ( Tensor2<double>(lMlh,vMlh) );
}

MI_PROPERTY(AlGaAsSb, Mh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence; "),
            MIComment("mh = (mhh^1.5+mlh^1.5)^(2/3)")
            )
Tensor2<double> AlGaAsSb::Mh(double T, double e) const {
    double tMc00 = pow(pow(Mhh(T,e).c00,1.5)+pow(Mlh(T,e).c00,1.5),(2./3.));
    double tMc11 = pow(pow(Mhh(T,e).c11,1.5)+pow(Mlh(T,e).c11,1.5),(2./3.));
    Tensor2<double> tMh(tMc00, tMc11); // [001]
    return ( tMh );
}

MI_PROPERTY(AlGaAsSb, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double AlGaAsSb::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(AlGaAsSb, VB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("nonlinear interpolation: GaSb, AlSb, GaAs, AlAs"),
            MIComment("no temperature dependence")
            )
double AlGaAsSb::VB(double T, double e, char point, char hole) const {
    double tVB( Ga*As*mGaAs.VB(T,0.,point,hole) + Ga*Sb*mGaSb.VB(T,0.,point,hole)
                + Al*As*mAlAs.VB(T,0.,point,hole) + Al*Sb*mAlSb.VB(T,0.,point,hole)
                - Al*As*Sb*(-1.71) - Ga*As*Sb*(-1.06) );
    if (!e) return tVB;
    else
    {
        double DEhy = 2.*av(T)*(1.-c12(T)/c11(T))*e;
        double DEsh = -2.*b(T)*(1.+2.*c12(T)/c11(T))*e;
        if (hole=='H') return ( tVB + DEhy - 0.5*DEsh );
        else if (hole=='L') return ( tVB + DEhy -0.5*Dso(T,e) + 0.25*DEsh + 0.5*sqrt(Dso(T,e)*Dso(T,e)+Dso(T,e)*DEsh+2.25*DEsh*DEsh) );
        else return 0.;
    }
}

MI_PROPERTY(AlGaAsSb, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaSb, AlSb, GaAs, AlAs"),
            MIComment("no temperature dependence")
            )
double AlGaAsSb::ac(double T) const {
    return ( Ga*As*mGaAs.ac(T) + Ga*Sb*mGaSb.ac(T)
             + Al*As*mAlAs.ac(T) + Al*Sb*mAlSb.ac(T) );
}

MI_PROPERTY(AlGaAsSb, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaSb, AlSb, GaAs, AlAs"),
            MIComment("no temperature dependence")
            )
double AlGaAsSb::av(double T) const {
    return ( Ga*As*mGaAs.av(T) + Ga*Sb*mGaSb.av(T) + Al*As*mAlAs.av(T) + Al*Sb*mAlSb.av(T) );
}

MI_PROPERTY(AlGaAsSb, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaSb, AlSb, GaAs, AlAs"),
            MIComment("no temperature dependence")
            )
double AlGaAsSb::b(double T) const {
    return ( Ga*As*mGaAs.b(T) + Ga*Sb*mGaSb.b(T) + Al*As*mAlAs.b(T) + Al*Sb*mAlSb.b(T) );
}

MI_PROPERTY(AlGaAsSb, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaSb, AlSb, GaAs, AlAs"),
            MIComment("no temperature dependence")
            )
double AlGaAsSb::d(double T) const {
    return ( Ga*As*mGaAs.d(T) + Ga*Sb*mGaSb.d(T) + Al*As*mAlAs.d(T) + Al*Sb*mAlSb.d(T) );
}

MI_PROPERTY(AlGaAsSb, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaSb, AlSb, GaAs, AlAs"),
            MIComment("no temperature dependence")
            )
double AlGaAsSb::c11(double T) const {
    return ( Ga*As*mGaAs.c11(T) + Ga*Sb*mGaSb.c11(T) + Al*As*mAlAs.c11(T) + Al*Sb*mAlSb.c11(T) );
}

MI_PROPERTY(AlGaAsSb, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaSb, AlSb, GaAs, AlAs"),
            MIComment("no temperature dependence")
            )
double AlGaAsSb::c12(double T) const {
    return ( Ga*As*mGaAs.c12(T) + Ga*Sb*mGaSb.c12(T) + Al*As*mAlAs.c12(T) + Al*Sb*mAlSb.c12(T) );
}

MI_PROPERTY(AlGaAsSb, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaSb, AlSb, GaAs, AlAs"),
            MIComment("no temperature dependence")
            )
double AlGaAsSb::c44(double T) const {
    return ( Ga*As*mGaAs.c44(T) + Ga*Sb*mGaSb.c44(T) + Al*As*mAlAs.c44(T) + Al*Sb*mAlSb.c44(T) );
}

MI_PROPERTY(AlGaAsSb, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.67; "), // 300 K
            MISource("S. Adachi, Properties of Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2005) p.37; "), // temperature dependence for binaries
            MISource("inversion od nonlinear interpolation of resistivity: GaSb, AlSb, GaAs, AlAs")
            )
Tensor2<double> AlGaAsSb::thermk(double T, double t) const {
    double lCondT = 1./(Ga*As/mGaAs.thermk(T,t).c00 + Ga*Sb/mGaSb.thermk(T,t).c00
                        + Al*As/mAlAs.thermk(T,t).c00 + Al*Sb/mAlSb.thermk(T,t).c00
                        + Al*Ga*0.32 + As*Sb*0.91),
           vCondT = 1./(Ga*As/mGaAs.thermk(T,t).c11 + Ga*Sb/mGaSb.thermk(T,t).c11
                        + Al*As/mAlAs.thermk(T,t).c11 + Al*Sb/mAlSb.thermk(T,t).c11
                        + Al*Ga*0.32 + As*Sb*0.91);
    return ( Tensor2<double>(lCondT,vCondT) );
}

MI_PROPERTY(AlGaAsSb, dens,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.18; "),
            MISource("linear interpolation: GaSb, AlSb, GaAs, AlAs"),
            MIComment("no temperature dependence")
            )
double AlGaAsSb::dens(double T) const {
    return ( Ga*As*mGaAs.dens(T) + Ga*Sb*mGaSb.dens(T) + Al*As*mAlAs.dens(T) + Al*Sb*mAlSb.dens(T) );
}

MI_PROPERTY(AlGaAsSb, cp,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.52; "),
            MISource("linear interpolation: GaSb, AlSb, GaAs, AlAs"),
            MIComment("no temperature dependence")
            )
double AlGaAsSb::cp(double T) const {
    return ( Ga*As*mGaAs.cp(T) + Ga*Sb*mGaSb.cp(T) + Al*As*mAlAs.cp(T) + Al*Sb*mAlSb.cp(T) );
}

MI_PROPERTY(AlGaAsSb, nr,
            MISource("C. Alibert et al., Journal of Applied Physics 69 (1991) 3208-3211")
            )
double AlGaAsSb::nr(double wl, double T, double n) const {
    double tE = phys::h_eVc1e9/wl; // wl -> E
    double tE0 = 1.89*Ga+3.2*Al-0.36*Al*Ga;
    double tEd = 24.5*Ga+28.*Al-4.4*Al*Ga;
    double tEG = 0.725*Ga+2.338*Al-0.47*Al*Ga;
    double nR300K2 = 1. + tEd/tE0 + tEd*tE*tE/pow(tE0,3.) + tEd*pow(tE,4.)/(2.*pow(tE0,3.)*(tE0*tE0-tEG*tEG)) * log((2.*tE0*tE0-tEG*tEG-tE*tE)/(tEG*tEG-tE*tE));

    double nR300K;
    if (nR300K2>0) nR300K = sqrt(nR300K2);
    else nR300K = 1.; // TODO
    double dnRdT = Al*As*4.6e-5 + Al*Sb*1.19e-5 + Ga*As*4.5e-5 + Ga*Sb*8.2e-5;
    return ( nR300K + nR300K*dnRdT*(T-300.) ); // 8.2e-5 - from Adachi (2005) ebook p.243 tab. 10.6
}

MI_PROPERTY(AlGaAsSb, absp,
            MIComment("TODO")
            )
double AlGaAsSb::absp(double wl, double T) const {
        throw NotImplemented("absp for AlGaAsSb");
}

bool AlGaAsSb::isEqual(const Material &other) const {
    const AlGaAsSb& o = static_cast<const AlGaAsSb&>(other);
    return o.Al == this->Al;
}

static MaterialsDB::Register<AlGaAsSb> materialDB_register_AlGaAsSb;

}} // namespace plask::materials
