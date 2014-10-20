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
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
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
            MISource("R. Kudrawiec, J. Appl. Phys. 101 (2007) 023522"),
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double GaInNAs::Eg(double T, double e, char point) const {
    double tEg(0.);
    if ((point == 'G') || (point == '*'))
    {
        point = 'G';
        double tEgG_GaNAs = 0.5 * ( /*GaNAs.En*/1.65 + mGaAs.Eg(T,e,point) - sqrt(pow(1.65-mGaAs.Eg(T,e,point),2.)+4./**GaNAs.V*/*2.7*2.7*N));
        double tEgG_InNAs = 0.5 * ( /*InNAs.En*/1.44 + mInAs.Eg(T,e,point) - sqrt(pow(1.44-mInAs.Eg(T,e,point),2.)+4./**InNAs.V*/*2.0*2.0*N));
        tEg = Ga*tEgG_GaNAs + In*tEgG_InNAs - Ga*In*(0.477);
    }
    else return 0.;
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}

MI_PROPERTY(GaInNAs, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MISource("nonlinear interpolation: GaNzb, InNzb, GaAs, InAs")
            )
double GaInNAs::Dso(double T, double e) const {
    return ( Ga*As*mGaAs.Dso(T,e) + Ga*N*mGaN.Dso(T,e)
             + In*As*mInAs.Dso(T,e) + In*N*mInN.Dso(T,e)
             - Ga*In*As*(0.15) /*- Ga*In*N*(0.) - Ga*As*N*(0.) - In*As*N*(0.)*/ );
}

MI_PROPERTY(GaInNAs, Me,
            MISource("Sarzala et al., Appl Phys A 108 (2012) 521-528"),
            MISource("nonlinear interpolation: GaNzb, InNzb, GaAs, InAs")
            )
Tensor2<double> GaInNAs::Me(double T, double e, char point) const {
    double lMe = Ga*As*mGaAs.Me(T,e,point).c00 + In*As*mInAs.Me(T,e,point).c00
            - Ga*In*(0.008) + 0.032,
           vMe = Ga*As*mGaAs.Me(T,e,point).c11 + In*As*mInAs.Me(T,e,point).c11
            - Ga*In*(0.008) + 0.032;
    return ( Tensor2<double>(lMe,vMe) );
}

MI_PROPERTY(GaInNAs, Mhh,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons, 2009"),
            MISource("linear interpolation: GaNzb, InNzb, GaAs, InAs")
            )
Tensor2<double> GaInNAs::Mhh(double T, double e) const {
    double lMhh = Ga*As*mGaAs.Mhh(T,e).c00 + Ga*N*mGaN.Mhh(T,e).c00
            + In*As*mInAs.Mhh(T,e).c00 + In*N*mInN.Mhh(T,e).c00,
           vMhh = Ga*As*mGaAs.Mhh(T,e).c11 + Ga*N*mGaN.Mhh(T,e).c11
            + In*As*mInAs.Mhh(T,e).c11 + In*N*mInN.Mhh(T,e).c11;
    return ( Tensor2<double>(lMhh,vMhh) );
}

MI_PROPERTY(GaInNAs, Mlh,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons, 2009"),
            MISource("linear interpolation: GaNzb, InNzb, GaAs, InAs")
            )
Tensor2<double> GaInNAs::Mlh(double T, double e) const {
    double lMlh = Ga*As*mGaAs.Mlh(T,e).c00 + Ga*N*mGaN.Mlh(T,e).c00
            + In*As*mInAs.Mlh(T,e).c00 + In*N*mInN.Mlh(T,e).c00,
           vMlh = Ga*As*mGaAs.Mlh(T,e).c11 + Ga*N*mGaN.Mlh(T,e).c11
            + In*As*mInAs.Mlh(T,e).c11 + In*N*mInN.Mlh(T,e).c11;
    return ( Tensor2<double>(lMlh,vMlh) );
}

MI_PROPERTY(GaInNAs, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696")
            )
double GaInNAs::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(GaInNAs, VB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MISource("linear interpolation: GaNzb, InNzb, GaAs, InAs")
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
        else return 0.;
    }
}

MI_PROPERTY(GaInNAs, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MISource("linear interpolation: GaNzb, InNzb, GaAs, InAs")
            )
double GaInNAs::ac(double T) const {
    return ( Ga*As*mGaAs.ac(T) + Ga*N*mGaN.ac(T)
             + In*As*mInAs.ac(T) + In*N*mInN.ac(T)
             - Ga*In*As*(2.61) /*- Ga*In*N*(0.) - Ga*As*N*(0.) - In*As*N*(0.)*/ );
}

MI_PROPERTY(GaInNAs, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MISource("linear interpolation: GaNzb, InNzb, GaAs, InAs")
            )
double GaInNAs::av(double T) const {
    return ( Ga*As*mGaAs.av(T) + Ga*N*mGaN.av(T) + In*As*mInAs.av(T) + In*N*mInN.av(T) );
}

MI_PROPERTY(GaInNAs, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MISource("linear interpolation: GaNzb, InNzb, GaAs, InAs")
            )
double GaInNAs::b(double T) const {
    return ( Ga*As*mGaAs.b(T) + Ga*N*mGaN.b(T) + In*As*mInAs.b(T) + In*N*mInN.b(T) );
}

MI_PROPERTY(GaInNAs, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MISource("linear interpolation: GaNzb, InNzb, GaAs, InAs")
            )
double GaInNAs::d(double T) const {
    return ( Ga*As*mGaAs.d(T) + Ga*N*mGaN.d(T) + In*As*mInAs.d(T) + In*N*mInN.d(T) );
}

MI_PROPERTY(GaInNAs, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MISource("linear interpolation: GaNzb, InNzb, GaAs, InAs")
            )
double GaInNAs::c11(double T) const {
    return ( Ga*As*mGaAs.c11(T) + Ga*N*mGaN.c11(T) + In*As*mInAs.c11(T) + In*N*mInN.c11(T) );
}

MI_PROPERTY(GaInNAs, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MISource("linear interpolation: GaNzb, InNzb, GaAs, InAs")
            )
double GaInNAs::c12(double T) const {
    return ( Ga*As*mGaAs.c12(T) + Ga*N*mGaN.c12(T) + In*As*mInAs.c12(T) + In*N*mInN.c12(T) );
}

MI_PROPERTY(GaInNAs, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MISource("linear interpolation: GaNzb, InNzb, GaAs, InAs")
            )
double GaInNAs::c44(double T) const {
    return ( Ga*As*mGaAs.c44(T) + Ga*N*mGaN.c44(T) + In*As*mInAs.c44(T) + In*N*mInN.c44(T) );
}

MI_PROPERTY(GaInNAs, thermk,
            MIComment("TODO")
            )
Tensor2<double> GaInNAs::thermk(double T, double t) const {
    return ( Tensor2<double>(0.,0.) );
}

MI_PROPERTY(GaInNAs, dens,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons, 2009"),
            MISource("linear interpolation: GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double GaInNAs::dens(double T) const {
    return ( Ga*As*mGaAs.dens(T) + Ga*N*mGaN.dens(T) + In*As*mInAs.dens(T) + In*N*mInN.dens(T) );
}

MI_PROPERTY(GaInNAs, cp,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons, 2009"),
            MISource("linear interpolation: GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double GaInNAs::cp(double T) const {
    return ( Ga*mGaAs.cp(T) + In*mInAs.cp(T) ); // till cp for GaN(zb) and InN(zb) unknown
}

MI_PROPERTY(GaInNAs, nr,
            MIComment("TODO")
            )
double GaInNAs::nr(double wl, double T, double n) const {
    return ( 0. );
}

MI_PROPERTY(GaInNAs, absp,
            MIComment("TODO")
            )
double GaInNAs::absp(double wl, double T) const {
    return ( 0. );
}

bool GaInNAs::isEqual(const Material &other) const {
    const GaInNAs& o = static_cast<const GaInNAs&>(other);
    return o.In == this->In;
}

static MaterialsDB::Register<GaInNAs> materialDB_register_GaInNAs;

}} // namespace plask::materials
