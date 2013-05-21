#include "InGaN.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string InGaN::name() const { return NAME; }

std::string InGaN::str() const { return StringBuilder("In", In)("Ga")("N"); }

InGaN::InGaN(const Material::Composition& Comp)
{
    In = Comp.find("In")->second;
    Ga = Comp.find("Ga")->second;
}

MI_PROPERTY(InGaN, thermk,
            MISource("B. N. Pantha et al., Applied Physics Letters 92 (2008) 042112"),
            MIComment("based on data for In: 16% - 36%")
            )
Tensor2<double> InGaN::thermk(double T, double t) const {
    double lCondT = 1/(In/mInN.thermk(T).c00 + Ga/mGaN.thermk(T,t).c00 + In*Ga*0.215*exp(7.913*In)),
           vCondT = 1/(In/mInN.thermk(T).c11 + Ga/mGaN.thermk(T,t).c11 + In*Ga*0.215*exp(7.913*In));
    return(Tensor2<double>(lCondT,vCondT));
 }

MI_PROPERTY(InGaN, absp,
            MISource("J. Piprek et al., Proc. SPIE 6766 (2007) 67660H"),
            MIComment("fit to GaN:Si/GaN:Mg/GaN:undoped in region 360 - 400 nm"),
            MIComment("no temperature dependence")
            )
double InGaN::absp(double wl, double T) const {
    double a = phys::h_eVc1e9/wl - Eg(T, 0, 'G');
    return ( 19000*exp(a/0.019) + 330*exp(a/0.07) );
}

MI_PROPERTY(InGaN, nr,
            MIComment("shift of the nR for GaN")
            )
double InGaN::nr(double wl, double T) const {
    double dEg = Eg(T,0.,'G') - mGaN.Eg(300.,0.,'G'),
           Eold = phys::h_eVc1e9 / wl,
           Enew = Eold - dEg;

    if (Enew > 1.000 && Enew < 2.138) // 580-1240 nm
        return ( 0.013914*Enew*Enew*Enew*Enew - 0.096422*Enew*Enew*Enew + 0.27318*Enew*Enew - 0.27725*Enew + 2.3535 );
    else if (Enew < 3.163) // 392-580 nm
        return ( 0.1152*Enew*Enew*Enew - 0.7955*Enew*Enew + 1.959*Enew + 0.68 );
    else if (Enew < 3.351) // 370-392 nm
        return ( 18.2292*Enew*Enew*Enew - 174.6974*Enew*Enew + 558.535*Enew - 593.164 );
    else if (Enew < 3.532) // 351-370 nm
        return ( 33.63905*Enew*Enew*Enew - 353.1446*Enew*Enew + 1235.0168*Enew - 1436.09 );
    else if (Enew < 4.100) // 336-351 nm
        return ( -0.72116*Enew*Enew*Enew + 8.8092*Enew*Enew - 35.8878*Enew + 51.335 );
    else if (Enew < 5.000) // 248-336 nm
        return ( 0.351664*Enew*Enew*Enew*Enew - 6.06337*Enew*Enew*Enew + 39.2317*Enew*Enew - 112.865*Enew + 124.358 );
    else
        return 0.;
}

MI_PROPERTY(InGaN, Eg,
            MISource("Vurgaftman et al. in Piprek 2007 Nitride Semicondcuctor Devices")
            )
double InGaN::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = In*mInN.Eg(T,e,point) + Ga*mGaN.Eg(T,e,point) - In*Ga*1.4;
    return (tEg);
}

MI_PROPERTY(InGaN, Me,
            MISource("linear interpolation: InN, GaN")
            )
Tensor2<double> InGaN::Me(double T, double e, char point) const {
    double lMe = In*mInN.Me(T,e,point).c00 + Ga*mGaN.Me(T,e,point).c00,
           vMe = In*mInN.Me(T,e,point).c11 + Ga*mGaN.Me(T,e,point).c11;
    return ( Tensor2<double>(lMe,vMe) );
}

MI_PROPERTY(InGaN, Mhh,
            MISource("linear interpolation: InN, GaN")
            )
Tensor2<double> InGaN::Mhh(double T, double e) const {
    double lMhh = In*mInN.Mhh(T,e).c00 + Ga*mGaN.Mhh(T,e).c00,
           vMhh = In*mInN.Mhh(T,e).c11 + Ga*mGaN.Mhh(T,e).c11;
    return ( Tensor2<double>(lMhh,vMhh) );
}

MI_PROPERTY(InGaN, Mlh,
            MISource("linear interpolation: InN, GaN")
            )
Tensor2<double> InGaN::Mlh(double T, double e) const {
    double lMlh = In*mInN.Mlh(T,e).c00 + Ga*mGaN.Mlh(T,e).c00,
           vMlh = In*mInN.Mlh(T,e).c11 + Ga*mGaN.Mlh(T,e).c11;
    return ( Tensor2<double>(lMlh,vMlh) );
}

MI_PROPERTY(InGaN, lattC,
            MISource("linear interpolation: GaN, InN")
            )
double InGaN::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = mInN.lattC(T,'a')*In + mGaN.lattC(T,'a')*Ga;
    else if (x == 'c') tLattC = mInN.lattC(T,'c')*In + mGaN.lattC(T,'c')*Ga;
    return (tLattC);
}

bool InGaN::isEqual(const Material &other) const {
    const InGaN& o = static_cast<const InGaN&>(other);
    return o.In == this->In;
}

static MaterialsDB::Register<InGaN> materialDB_register_InGaN;

}       // namespace plask

