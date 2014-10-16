#include "AlInAs.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

AlInAs::AlInAs(const Material::Composition& Comp) {
    Al = Comp.find("Al")->second;
    In = Comp.find("In")->second;
}

std::string AlInAs::str() const { return StringBuilder("Al", Al)("In")("As"); }

std::string AlInAs::name() const { return NAME; }

MI_PROPERTY(AlInAs, lattC,
            MISource("linear interpolation: AlAs, InAs")
            )
double AlInAs::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = Al*mAlAs.lattC(T,'a') + In*mInAs.lattC(T,'a');
    else if (x == 'c') tLattC = Al*mAlAs.lattC(T,'c') + In*mInAs.lattC(T,'c');
    return ( tLattC );
}

MI_PROPERTY(AlInAs, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double AlInAs::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = Al*mAlAs.Eg(T,e,point) + In*mInAs.Eg(T,e,point) - Al*In*(0.70);
    else if (point == 'X') tEg = Al*mAlAs.Eg(T,e,point) + In*mInAs.Eg(T,e,point);
    else if (point == 'L') tEg = Al*mAlAs.Eg(T,e,point) + In*mInAs.Eg(T,e,point);
    else if (point == '*')
    {
        double tEgG = Al*mAlAs.Eg(T,e,point) + In*mInAs.Eg(T,e,point) - Al*In*(0.70);
        double tEgX = Al*mAlAs.Eg(T,e,point) + In*mInAs.Eg(T,e,point);
        double tEgL = Al*mAlAs.Eg(T,e,point) + In*mInAs.Eg(T,e,point);
        tEg = min(tEgG,min(tEgX,tEgL));
    }
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}

MI_PROPERTY(AlInAs, Dso,
            MISource("nonlinear interpolation: AlAs, InAs")
            )
double AlInAs::Dso(double T, double e) const {
    return ( Al*mAlAs.Dso(T,e) + In*mInAs.Dso(T,e) - Al*In*(0.15) );
}

MI_PROPERTY(AlInAs, Me,
            MISource("nonlinear interpolation: AlAs, InAs")
            )
Tensor2<double> AlInAs::Me(double T, double e, char point) const {
    double lMe = Al*mAlAs.Me(T,e,point).c00 + In*mInAs.Me(T,e,point).c00 - Al*In*(0.012),
           vMe = Al*mAlAs.Me(T,e,point).c11 + In*mInAs.Me(T,e,point).c11 - Al*In*(0.012);
    return ( Tensor2<double>(lMe,vMe) );
}

MI_PROPERTY(AlInAs, Mhh,
            MISource("linear interpolation: AlAs, InAs")
            )
Tensor2<double> AlInAs::Mhh(double T, double e) const {
    double lMhh = Al*mAlAs.Mhh(T,e).c00 + In*mInAs.Mhh(T,e).c00,
           vMhh = Al*mAlAs.Mhh(T,e).c11 + In*mInAs.Mhh(T,e).c11;
    return ( Tensor2<double>(lMhh,vMhh) );
}

MI_PROPERTY(AlInAs, Mlh,
            MISource("linear interpolation: AlAs, InAs")
            )
Tensor2<double> AlInAs::Mlh(double T, double e) const {
    double lMlh = Al*mAlAs.Mlh(T,e).c00 + In*mInAs.Mlh(T,e).c00,
           vMlh = Al*mAlAs.Mlh(T,e).c11 + In*mInAs.Mlh(T,e).c11;
    return ( Tensor2<double>(lMlh,vMlh) );
}

MI_PROPERTY(AlInAs, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double AlInAs::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(AlInAs, VB,
            MISource("linear interpolation: AlAs, InAs")
            )
double AlInAs::VB(double T, double e, char point, char hole) const {
    double tVB( Al*mAlAs.VB(T,0.,point,hole) + In*mInAs.VB(T,0.,point,hole) );
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

MI_PROPERTY(AlInAs, ac,
            MISource("linear interpolation: AlAs, InAs")
            )
double AlInAs::ac(double T) const {
    return ( Al*mAlAs.ac(T) + In*mInAs.ac(T) );
}

MI_PROPERTY(AlInAs, av,
            MISource("linear interpolation: AlAs, InAs")
            )
double AlInAs::av(double T) const {
    return ( Al*mAlAs.av(T) + In*mInAs.av(T) );
}

MI_PROPERTY(AlInAs, b,
            MISource("linear interpolation: AlAs, InAs")
            )
double AlInAs::b(double T) const {
    return ( Al*mAlAs.b(T) + In*mInAs.b(T) );
}

MI_PROPERTY(AlInAs, d,
            MISource("linear interpolation: AlAs, InAs")
            )
double AlInAs::d(double T) const {
    return ( Al*mAlAs.d(T) + In*mInAs.d(T) );
}

MI_PROPERTY(AlInAs, c11,
            MISource("linear interpolation: AlAs, InAs")
            )
double AlInAs::c11(double T) const {
    return ( Al*mAlAs.c11(T) + In*mInAs.c11(T) );
}

MI_PROPERTY(AlInAs, c12,
            MISource("linear interpolation: AlAs, InAs")
            )
double AlInAs::c12(double T) const {
    return ( Al*mAlAs.c12(T) + In*mInAs.c12(T) );
}

MI_PROPERTY(AlInAs, c44,
            MISource("linear interpolation: AlAs, InAs")
            )
double AlInAs::c44(double T) const {
    return ( Al*mAlAs.c44(T) + In*mInAs.c44(T) );
}

MI_PROPERTY(AlInAs, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009")
            )
Tensor2<double> AlInAs::thermk(double T, double t) const {
    double lCondT = 1./(Al/mAlAs.thermk(T,t).c00 + In/mInAs.thermk(T,t).c00 + Al*In*0.15),
           vCondT = 1./(Al/mAlAs.thermk(T,t).c11 + In/mInAs.thermk(T,t).c11 + Al*In*0.15);
    return ( Tensor2<double>(lCondT,vCondT) );
}

MI_PROPERTY(AlInAs, nr,
            MISource("M.J. Mondry et al., IEEE Photon. Technol. Lett. 4 (1992) 627-630"),
            MIComment("data for the wavelength ranging 1000-2000 nm")
            )
double AlInAs::nr(double wl, double T, double n) const {
    double tnr = sqrt(8.677 + (1.214*wl*wl)/(wl*wl-730.8*730.8)), //wl: 1000-2000 nm
           tBeta = 3.5e-4; //D. Dey for In = 0.365
    return ( tnr + tBeta*(T-300.) );
}

MI_PROPERTY(AlInAs, absp,
            MIComment("TODO")
            )
double AlInAs::absp(double wl, double T) const {
    return ( 0. );
}

bool AlInAs::isEqual(const Material &other) const {
    const AlInAs& o = static_cast<const AlInAs&>(other);
    return o.Al == this->Al;
}

static MaterialsDB::Register<AlInAs> materialDB_register_AlInAs;

}} // namespace plask::materials
