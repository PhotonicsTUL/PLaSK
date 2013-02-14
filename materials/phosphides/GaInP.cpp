#include "GaInP.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

GaInP::GaInP(const Material::Composition& Comp) {
    Ga = Comp.find("Ga")->second;
    In = Comp.find("In")->second;
}

std::string GaInP::str() const { return StringBuilder("Ga")("In", In)("P"); }

std::string GaInP::name() const { return NAME; }

MI_PROPERTY(GaInP, lattC,
            MISource("linear interpolation: GaP, InP")
            )
double GaInP::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = Ga*mGaP.lattC(T,'a') + In*mInP.lattC(T,'a');
    else if (x == 'c') tLattC = Ga*mGaP.lattC(T,'a') + In*mInP.lattC(T,'a');
    return ( tLattC );
}

MI_PROPERTY(GaInP, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("only for Gamma point")
            )
double GaInP::Eg(double T, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = Ga*mGaP.Eg(T, point) + In*mInP.Eg(T, point) - Ga*In*0.65;
    return ( tEg );
}

MI_PROPERTY(GaInP, Dso,
            MISource("linear interpolation: GaP, InP")
            )
double GaInP::Dso(double T) const {
    return ( Ga*mGaP.Dso(T) + In*mInP.Dso(T) );
}

MI_PROPERTY(GaInP, Me,
            MISource("nonlinear interpolation: GaP, InP")
            )
Tensor2<double> GaInP::Me(double T, char point) const {
    double lMe = Ga*mGaP.Me(T,point).c00 + In*mInP.Me(T,point).c00 - Ga*In*0.01854,
           vMe = Ga*mGaP.Me(T,point).c11 + In*mInP.Me(T,point).c11 - Ga*In*0.01854;
    return ( Tensor2<double>(lMe,vMe) );
}

MI_PROPERTY(GaInP, Mhh,
            MISource("linear interpolation: GaP, InP")
            )
Tensor2<double> GaInP::Mhh(double T, char point) const {
    double lMhh = Ga*mGaP.Mhh(T,point).c00 + In*mInP.Mhh(T,point).c00,
           vMhh = Ga*mGaP.Mhh(T,point).c11 + In*mInP.Mhh(T,point).c11;
    return ( Tensor2<double>(lMhh,vMhh) );
}

MI_PROPERTY(GaInP, Mlh,
            MISource("linear interpolation: GaP, InP")
            )
Tensor2<double> GaInP::Mlh(double T, char point) const {
    double lMlh = Ga*mGaP.Mlh(T,point).c00 + In*mInP.Mlh(T,point).c00,
           vMlh = Ga*mGaP.Mlh(T,point).c11 + In*mInP.Mlh(T,point).c11;
    return ( Tensor2<double>(lMlh,vMlh) );
}

MI_PROPERTY(GaInP, ac,
            MISource("linear interpolation: GaP, InP")
            )
double GaInP::ac(double T) const {
    return ( Ga*mGaP.ac(T) + In*mInP.ac(T) );
}

MI_PROPERTY(GaInP, av,
            MISource("linear interpolation: GaP, InP")
            )
double GaInP::av(double T) const {
    return ( Ga*mGaP.av(T) + In*mInP.av(T) );
}

MI_PROPERTY(GaInP, b,
            MISource("linear interpolation: GaP, InP")
            )
double GaInP::b(double T) const {
    return ( Ga*mGaP.b(T) + In*mInP.b(T) );
}

MI_PROPERTY(GaInP, c11,
            MISource("linear interpolation: GaP, InP")
            )
double GaInP::c11(double T) const {
    return ( Ga*mGaP.c11(T) + In*mInP.c11(T) );
}

MI_PROPERTY(GaInP, c12,
            MISource("linear interpolation: GaP, InP")
            )
double GaInP::c12(double T) const {
    return ( Ga*mGaP.c12(T) + In*mInP.c12(T) );
}

MI_PROPERTY(GaInP, A,
            MISource("L. Piskorski, master thesis"),
            MIComment("no temperature dependence")
            )
double GaInP::A(double T) const {
    return ( 1e8 );
}

MI_PROPERTY(GaInP, B,
            MISource("L. Piskorski, master thesis"),
            MIComment("TODO")
            )
double GaInP::B(double T) const {
    return ( 1e-10*pow(300/T,1.5) );
}

MI_PROPERTY(GaInP, C,
            MISource("W. W. Chow et al., IEEE Journal of Selected Topics in Quantum Electronics 1 (1995) 649-653"),
            MIComment("no temperature dependence"),
            MIComment("TODO")
            )
double GaInP::C(double T) const {
    return ( 3.5e-30 );
}

MI_PROPERTY(GaInP, D,
            MISource("O. Imafuji et al., Journal of Selected Topics in Quantum Electronics 5 (1999) 721-728"), // D(300K)
            MISource("L. Piskorski, master thesis") // D(T)
            )
double GaInP::D(double T) const {
    return ( T/300. );
}

MI_PROPERTY(GaInP, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009")
            )
Tensor2<double> GaInP::thermk(double T, double t) const {
    double lCondT = 1./(Ga/mGaP.thermk(T,t).c00 + In/mInP.thermk(T,t).c00 + Ga*In*0.72),
           vCondT = 1./(Ga/mGaP.thermk(T,t).c11 + In/mInP.thermk(T,t).c11 + Ga*In*0.72);
    return ( Tensor2<double>(lCondT,vCondT) );
}

MI_PROPERTY(GaInP, nr,
            MIComment("TODO")
            )
double GaInP::nr(double wl, double T) const {
    return ( 0. );
}

MI_PROPERTY(GaInP, absp,
            MIComment("TODO")
            )
double GaInP::absp(double wl, double T) const {
    return ( 0. );
}

bool GaInP::isEqual(const Material &other) const {
    const GaInP& o = static_cast<const GaInP&>(other);
    return o.Ga == this->Ga;
}

static MaterialsDB::Register<GaInP> materialDB_register_GaInP;

} // namespace plask
