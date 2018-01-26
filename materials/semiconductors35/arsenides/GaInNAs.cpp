#include "GaInNAs.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

GaInNAs::GaInNAs(const Material::Composition& Comp) {
    Ga = Comp.find("Ga")->second;
    In = Comp.find("In")->second;
    N = Comp.find("N")->second;
    As = Comp.find("As")->second;
}

std::string GaInNAs::str() const { return StringBuilder("Ga")("In", In)("N", N)("As"); }

std::string GaInNAs::name() const { return NAME; }

MI_PROPERTY(GaInNAs, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696; "),
            MISource("linear interpolation: GaNzb, InNzb, GaAs, InAs")
            )
double GaInNAs::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = Ga*As*mGaAs.lattC(T,'a') + Ga*N*mGaN.lattC(T,'a')
            + In*As*mInAs.lattC(T,'a') + In*N*mInN.lattC(T,'a');
    else if (x == 'c') tLattC = Ga*As*mGaAs.lattC(T,'c') + Ga*N*mGaN.lattC(T,'c')
            + In*As*mInAs.lattC(T,'c') + In*N*mInN.lattC(T,'c');
    return ( tLattC );
}

MI_PROPERTY(GaInNAs, Eg,
            MISource("R. Kudrawiec, J. Appl. Phys. 101 (2007) 023522; "),
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696; "),
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("only for Gamma point")
            )
double GaInNAs::Eg(double T, double e, char point) const {
    double tEg(0.);
    if ((point == 'G') || (point == '*')) {
        point = 'G';
        double tEgG_GaNAs = 0.5 * ( /*GaNAs.En*/1.65 + mGaAs.Eg(T,e,point) - sqrt(pow(1.65-mGaAs.Eg(T,e,point),2.)+4./**GaNAs.V*/*2.7*2.7*N));
        double tEgG_InNAs = 0.5 * ( /*InNAs.En*/1.44 + mInAs.Eg(T,e,point) - sqrt(pow(1.44-mInAs.Eg(T,e,point),2.)+4./**InNAs.V*/*2.0*2.0*N));
        tEg = Ga*tEgG_GaNAs + In*tEgG_InNAs - Ga*In*(0.477);
    }
    else throw NotImplemented("EgX and EgL for GaInNAs");
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}

MI_PROPERTY(GaInNAs, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696; "),
            MISource("nonlinear interpolation: GaNzb, InNzb, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double GaInNAs::Dso(double T, double e) const {
    return ( Ga*As*mGaAs.Dso(T,e) + Ga*N*mGaN.Dso(T,e)
             + In*As*mInAs.Dso(T,e) + In*N*mInN.Dso(T,e)
             - Ga*In*As*(0.15) /*- Ga*In*N*(0.) - Ga*As*N*(0.) - In*As*N*(0.)*/ );
}

MI_PROPERTY(GaInNAs, Me,
            MISource("Sarzala et al., Appl Phys A 108 (2012) 521-528; "), // bowing parameter
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.230-232; "),
            MISource("nonlinear interpolation: GaNzb, InNzb, GaAs, InAs"),
            MIComment("only for Gamma point; "),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaInNAs::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0., 0.);
    if ((point == 'G') || (point == '*')) {
        tMe.c00 = Ga*mGaAs.Me(T,e,point).c00 + In*mInAs.Me(T,e,point).c00;
        tMe.c11 = Ga*mGaAs.Me(T,e,point).c11 + In*mInAs.Me(T,e,point).c11;

        tMe.c00 += ( -Ga*In*(0.008)+0.032 );
        tMe.c11 += ( -Ga*In*(0.008)+0.032 );
    }
    else throw NotImplemented("Me for X and L points for GaInNAs");
    return ( tMe );
}

MI_PROPERTY(GaInNAs, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235; "),
            MISource("linear interpolation: GaNzb, InNzb, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaInNAs::Mhh(double T, double e) const {
    double lMhh = Ga*As*mGaAs.Mhh(T,e).c00 + Ga*N*mGaN.Mhh(T,e).c00
            + In*As*mInAs.Mhh(T,e).c00 + In*N*mInN.Mhh(T,e).c00,
           vMhh = Ga*As*mGaAs.Mhh(T,e).c11 + Ga*N*mGaN.Mhh(T,e).c11
            + In*As*mInAs.Mhh(T,e).c11 + In*N*mInN.Mhh(T,e).c11;
    return ( Tensor2<double>(lMhh,vMhh) );
}

MI_PROPERTY(GaInNAs, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235; "),
            MISource("linear interpolation: GaNzb, InNzb, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaInNAs::Mlh(double T, double e) const {
    double lMlh = Ga*As*mGaAs.Mlh(T,e).c00 + Ga*N*mGaN.Mlh(T,e).c00
            + In*As*mInAs.Mlh(T,e).c00 + In*N*mInN.Mlh(T,e).c00,
           vMlh = Ga*As*mGaAs.Mlh(T,e).c11 + Ga*N*mGaN.Mlh(T,e).c11
            + In*As*mInAs.Mlh(T,e).c11 + In*N*mInN.Mlh(T,e).c11;
    return ( Tensor2<double>(lMlh,vMlh) );
}

MI_PROPERTY(GaInNAs, Mh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence; "),
            MIComment("mh = (mhh^1.5+mlh^1.5)^(2/3)")
            )
Tensor2<double> GaInNAs::Mh(double T, double e) const {
    double tMc00 = pow(pow(Mhh(T,e).c00,1.5)+pow(Mlh(T,e).c00,1.5),(2./3.));
    double tMc11 = pow(pow(Mhh(T,e).c11,1.5)+pow(Mlh(T,e).c11,1.5),(2./3.));
    Tensor2<double> tMh(tMc00, tMc11); // [001]
    return ( tMh );
}

MI_PROPERTY(GaInNAs, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MIComment("only for Gamma point")
            )
double GaInNAs::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(GaInNAs, VB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696; "),
            MISource("linear interpolation: GaNzb, InNzb, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double GaInNAs::VB(double T, double e, char point, char hole) const {
    double tVB( Ga*As*mGaAs.VB(T,0.,point,hole) + Ga*N*mGaN.VB(T,0.,point,hole)
                + In*As*mInAs.VB(T,0.,point,hole) + In*N*mInN.VB(T,0.,point,hole)
                - Ga*In*As*(-0.38) /*- Ga*In*N*(0.) - Ga*As*N*(0.) - In*As*N*(0.)*/ );
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

MI_PROPERTY(GaInNAs, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696; "),
            MISource("linear interpolation: GaNzb, InNzb, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double GaInNAs::ac(double T) const {
    return ( Ga*As*mGaAs.ac(T) + Ga*N*mGaN.ac(T)
             + In*As*mInAs.ac(T) + In*N*mInN.ac(T)
             - Ga*In*As*(2.61) /*- Ga*In*N*(0.) - Ga*As*N*(0.) - In*As*N*(0.)*/ );
}

MI_PROPERTY(GaInNAs, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696; "),
            MISource("linear interpolation: GaNzb, InNzb, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double GaInNAs::av(double T) const {
    return ( Ga*As*mGaAs.av(T) + Ga*N*mGaN.av(T) + In*As*mInAs.av(T) + In*N*mInN.av(T) );
}

MI_PROPERTY(GaInNAs, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696; "),
            MISource("linear interpolation: GaNzb, InNzb, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double GaInNAs::b(double T) const {
    return ( Ga*As*mGaAs.b(T) + Ga*N*mGaN.b(T) + In*As*mInAs.b(T) + In*N*mInN.b(T) );
}

MI_PROPERTY(GaInNAs, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696; "),
            MISource("linear interpolation: GaNzb, InNzb, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double GaInNAs::d(double T) const {
    return ( Ga*As*mGaAs.d(T) + Ga*N*mGaN.d(T) + In*As*mInAs.d(T) + In*N*mInN.d(T) );
}

MI_PROPERTY(GaInNAs, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696; "),
            MISource("linear interpolation: GaNzb, InNzb, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double GaInNAs::c11(double T) const {
    return ( Ga*As*mGaAs.c11(T) + Ga*N*mGaN.c11(T) + In*As*mInAs.c11(T) + In*N*mInN.c11(T) );
}

MI_PROPERTY(GaInNAs, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696; "),
            MISource("linear interpolation: GaNzb, InNzb, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double GaInNAs::c12(double T) const {
    return ( Ga*As*mGaAs.c12(T) + Ga*N*mGaN.c12(T) + In*As*mInAs.c12(T) + In*N*mInN.c12(T) );
}

MI_PROPERTY(GaInNAs, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696; "),
            MISource("linear interpolation: GaNzb, InNzb, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double GaInNAs::c44(double T) const {
    return ( Ga*As*mGaAs.c44(T) + Ga*N*mGaN.c44(T) + In*As*mInAs.c44(T) + In*N*mInN.c44(T) );
}

MI_PROPERTY(GaInNAs, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.67; "), // 300 K
            MISource("S. Adachi, Properties of Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2005) p.37; "), // temperature dependence for binaries
            MISource("inversion of nonlinear interpolation of resistivity: GaAs, InAs"),
            MIComment("no influence of nitrogen")
            )
Tensor2<double> GaInNAs::thermk(double T, double t) const {
    double lCondT = 1./(Ga/mGaAs.thermk(T,t).c00 + In/mInAs.thermk(T,t).c00 + Ga*In*0.72),
           vCondT = 1./(Ga/mGaAs.thermk(T,t).c11 + In/mInAs.thermk(T,t).c11 + Ga*In*0.72);
    return ( Tensor2<double>(lCondT,vCondT) );
}

MI_PROPERTY(GaInNAs, dens,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.18; "),
            MISource("linear interpolation: GaNzb, InNzb, GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double GaInNAs::dens(double T) const {
    return ( Ga*As*mGaAs.dens(T) + Ga*N*mGaN.dens(T) + In*As*mInAs.dens(T) + In*N*mInN.dens(T) );
}

MI_PROPERTY(GaInNAs, cp,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.52; "),
            MISource("linear interpolation: GaAs, InAs"),
            MIComment("no influence of nitrogen; "),
            MIComment("no temperature dependence")
            )
double GaInNAs::cp(double T) const {
    return ( Ga*mGaAs.cp(T) + In*mInAs.cp(T) ); // till cp for GaN(zb) and InN(zb) unknown
}

Material::ConductivityType GaInNAs::condtype() const { return Material::CONDUCTIVITY_I; }

MI_PROPERTY(GaInNAs, nr,
            MIComment("TODO")
            )
double GaInNAs::nr(double /*lam*/, double /*T*/, double /*n*/) const {
    throw NotImplemented("nr for GaInNAs");
}

MI_PROPERTY(GaInNAs, absp,
            MIComment("TODO")
            )
double GaInNAs::absp(double /*lam*/, double /*T*/) const {
    throw NotImplemented("absp for GaInNAs");
}

bool GaInNAs::isEqual(const Material &other) const {
    const GaInNAs& o = static_cast<const GaInNAs&>(other);
    return o.In == this->In;
}

static MaterialsDB::Register<GaInNAs> materialDB_register_GaInNAs;

}} // namespace plask::materials
