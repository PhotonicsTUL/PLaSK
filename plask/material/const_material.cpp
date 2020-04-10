#include "const_material.h"

namespace plask {

ConstMaterial::ConstMaterial(const std::string& definition) {
    
    for (auto item: boost::tokenizer<boost::char_separator<char>>(definition, boost::char_separator<char>(" ,"))) {
        std::string key, value;
        std::tie(key, value) = splitString2(item, '=');
        try {
            if (key.empty()) continue;
            else if (key == "lattC") cache.lattC.reset(boost::lexical_cast<double>(value));
            else if (key == "Eg") cache.Eg.reset(boost::lexical_cast<double>(value));
            else if (key == "CB") cache.CB.reset(boost::lexical_cast<double>(value));
            else if (key == "VB") cache.VB.reset(boost::lexical_cast<double>(value));
            else if (key == "Dso") cache.Dso.reset(boost::lexical_cast<double>(value));
            else if (key == "Mso") cache.Mso.reset(boost::lexical_cast<double>(value));
            else if (key == "Me") cache.Me.reset(boost::lexical_cast<double>(value));
            else if (key == "Mhh") cache.Mhh.reset(boost::lexical_cast<double>(value));
            else if (key == "Mlh") cache.Mlh.reset(boost::lexical_cast<double>(value));
            else if (key == "Mh") cache.Mh.reset(boost::lexical_cast<double>(value));
            else if (key == "y1") cache.y1.reset(boost::lexical_cast<double>(value));
            else if (key == "y2") cache.y2.reset(boost::lexical_cast<double>(value));
            else if (key == "y3") cache.y3.reset(boost::lexical_cast<double>(value));
            else if (key == "ac") cache.ac.reset(boost::lexical_cast<double>(value));
            else if (key == "av") cache.av.reset(boost::lexical_cast<double>(value));
            else if (key == "b") cache.b.reset(boost::lexical_cast<double>(value));
            else if (key == "d") cache.d.reset(boost::lexical_cast<double>(value));
            else if (key == "c11") cache.c11.reset(boost::lexical_cast<double>(value));
            else if (key == "c12") cache.c12.reset(boost::lexical_cast<double>(value));
            else if (key == "c44") cache.c44.reset(boost::lexical_cast<double>(value));
            else if (key == "eps") cache.eps.reset(boost::lexical_cast<double>(value));
            else if (key == "chi") cache.chi.reset(boost::lexical_cast<double>(value));
            else if (key == "Ni") cache.Ni.reset(boost::lexical_cast<double>(value));
            else if (key == "Nf") cache.Nf.reset(boost::lexical_cast<double>(value));
            else if (key == "EactD") cache.EactD.reset(boost::lexical_cast<double>(value));
            else if (key == "EactA") cache.EactA.reset(boost::lexical_cast<double>(value));
            else if (key == "mob") cache.mob.reset(boost::lexical_cast<double>(value));
            else if (key == "cond") cache.cond.reset(boost::lexical_cast<double>(value));
            else if (key == "A") cache.A.reset(boost::lexical_cast<double>(value));
            else if (key == "B") cache.B.reset(boost::lexical_cast<double>(value));
            else if (key == "C") cache.C.reset(boost::lexical_cast<double>(value));
            else if (key == "D") cache.D.reset(boost::lexical_cast<double>(value));
            else if (key == "thermk") cache.thermk.reset(boost::lexical_cast<double>(value));
            else if (key == "dens") cache.dens.reset(boost::lexical_cast<double>(value));
            else if (key == "cp") cache.cp.reset(boost::lexical_cast<double>(value));
            else if (key == "nr") cache.nr.reset(boost::lexical_cast<double>(value));
            else if (key == "absp") cache.absp.reset(boost::lexical_cast<double>(value));
            else if (key == "mobe") cache.mobe.reset(boost::lexical_cast<double>(value));
            else if (key == "mobh") cache.mobh.reset(boost::lexical_cast<double>(value));
            else if (key == "taue") cache.taue.reset(boost::lexical_cast<double>(value));
            else if (key == "tauh") cache.tauh.reset(boost::lexical_cast<double>(value));
            else if (key == "Ce") cache.Ce.reset(boost::lexical_cast<double>(value));
            else if (key == "Ch") cache.Ch.reset(boost::lexical_cast<double>(value));
            else if (key == "e13") cache.e13.reset(boost::lexical_cast<double>(value));
            else if (key == "e15") cache.e15.reset(boost::lexical_cast<double>(value));
            else if (key == "e33") cache.e33.reset(boost::lexical_cast<double>(value));
            else if (key == "c13") cache.c13.reset(boost::lexical_cast<double>(value));
            else if (key == "c33") cache.c33.reset(boost::lexical_cast<double>(value));
            else if (key == "Psp") cache.Psp.reset(boost::lexical_cast<double>(value));
            else if (key == "Na") cache.Na.reset(boost::lexical_cast<double>(value));
            else if (key == "Nd") cache.Nd.reset(boost::lexical_cast<double>(value));
            else throw MaterialParseException("({}): Bad material parameter '{}'", definition, key);
        } catch (boost::bad_lexical_cast&) {
            throw MaterialParseException("({}): Bad material parameter value '{}={}'", definition, key, value);
        }
    }

}

ConstMaterial::ConstMaterial(const std::map<std::string, double>& items) {
    for (auto item: items) {
        if (item.first == "lattC") cache.lattC.reset(item.second);
        else if (item.first == "Eg") cache.Eg.reset(item.second);
        else if (item.first == "CB") cache.CB.reset(item.second);
        else if (item.first == "VB") cache.VB.reset(item.second);
        else if (item.first == "Dso") cache.Dso.reset(item.second);
        else if (item.first == "Mso") cache.Mso.reset(item.second);
        else if (item.first == "Me") cache.Me.reset(item.second);
        else if (item.first == "Mhh") cache.Mhh.reset(item.second);
        else if (item.first == "Mlh") cache.Mlh.reset(item.second);
        else if (item.first == "Mh") cache.Mh.reset(item.second);
        else if (item.first == "y1") cache.y1.reset(item.second);
        else if (item.first == "y2") cache.y2.reset(item.second);
        else if (item.first == "y3") cache.y3.reset(item.second);
        else if (item.first == "ac") cache.ac.reset(item.second);
        else if (item.first == "av") cache.av.reset(item.second);
        else if (item.first == "b") cache.b.reset(item.second);
        else if (item.first == "d") cache.d.reset(item.second);
        else if (item.first == "c11") cache.c11.reset(item.second);
        else if (item.first == "c12") cache.c12.reset(item.second);
        else if (item.first == "c44") cache.c44.reset(item.second);
        else if (item.first == "eps") cache.eps.reset(item.second);
        else if (item.first == "chi") cache.chi.reset(item.second);
        else if (item.first == "Ni") cache.Ni.reset(item.second);
        else if (item.first == "Nf") cache.Nf.reset(item.second);
        else if (item.first == "EactD") cache.EactD.reset(item.second);
        else if (item.first == "EactA") cache.EactA.reset(item.second);
        else if (item.first == "mob") cache.mob.reset(item.second);
        else if (item.first == "cond") cache.cond.reset(item.second);
        else if (item.first == "A") cache.A.reset(item.second);
        else if (item.first == "B") cache.B.reset(item.second);
        else if (item.first == "C") cache.C.reset(item.second);
        else if (item.first == "D") cache.D.reset(item.second);
        else if (item.first == "thermk") cache.thermk.reset(item.second);
        else if (item.first == "dens") cache.dens.reset(item.second);
        else if (item.first == "cp") cache.cp.reset(item.second);
        else if (item.first == "nr") cache.nr.reset(item.second);
        else if (item.first == "absp") cache.absp.reset(item.second);
        else if (item.first == "mobe") cache.mobe.reset(item.second);
        else if (item.first == "mobh") cache.mobh.reset(item.second);
        else if (item.first == "taue") cache.taue.reset(item.second);
        else if (item.first == "tauh") cache.tauh.reset(item.second);
        else if (item.first == "Ce") cache.Ce.reset(item.second);
        else if (item.first == "Ch") cache.Ch.reset(item.second);
        else if (item.first == "e13") cache.e13.reset(item.second);
        else if (item.first == "e15") cache.e15.reset(item.second);
        else if (item.first == "e33") cache.e33.reset(item.second);
        else if (item.first == "c13") cache.c13.reset(item.second);
        else if (item.first == "c33") cache.c33.reset(item.second);
        else if (item.first == "Psp") cache.Psp.reset(item.second);
        else if (item.first == "Na") cache.Na.reset(item.second);
        else if (item.first == "Nd") cache.Nd.reset(item.second);
        else throw MaterialParseException("Bad material parameter '{}'", item.first);
    }
}


std::string ConstMaterial::str() const{
    std::string result;
    bool c = false;
    if (cache.A) { result += (c? ",A=" : "A=") + plask::str(*cache.A); c= true; }
    if (cache.absp) { result += (c? ",absp=" : "absp=") + plask::str(*cache.absp); c= true; }
    if (cache.ac) { result += (c? ",ac=" : "ac=") + plask::str(*cache.ac); c= true; }
    if (cache.av) { result += (c? ",av=" : "av=") + plask::str(*cache.av); c= true; }
    if (cache.b) { result += (c? ",b=" : "b=") + plask::str(*cache.b); c= true; }
    if (cache.B) { result += (c? ",B=" : "B=") + plask::str(*cache.B); c= true; }
    if (cache.C) { result += (c? ",C=" : "C=") + plask::str(*cache.C); c= true; }
    if (cache.c11) { result += (c? ",c11=" : "c11=") + plask::str(*cache.c11); c= true; }
    if (cache.c12) { result += (c? ",c12=" : "c12=") + plask::str(*cache.c12); c= true; }
    if (cache.c13) { result += (c? ",c13=" : "c13=") + plask::str(*cache.c13); c= true; }
    if (cache.c33) { result += (c? ",c33=" : "c33=") + plask::str(*cache.c33); c= true; }
    if (cache.c44) { result += (c? ",c44=" : "c44=") + plask::str(*cache.c44); c= true; }
    if (cache.CB) { result += (c? ",CB=" : "CB=") + plask::str(*cache.CB); c= true; }
    if (cache.Ce) { result += (c? ",Ce=" : "Ce=") + plask::str(*cache.Ce); c= true; }
    if (cache.Ch) { result += (c? ",Ch=" : "Ch=") + plask::str(*cache.Ch); c= true; }
    if (cache.chi) { result += (c? ",chi=" : "chi=") + plask::str(*cache.chi); c= true; }
    if (cache.cond) { result += (c? ",cond=" : "cond=") + plask::str(*cache.cond); c= true; }
    if (cache.cp) { result += (c? ",cp=" : "cp=") + plask::str(*cache.cp); c= true; }
    if (cache.d) { result += (c? ",d=" : "d=") + plask::str(*cache.d); c= true; }
    if (cache.D) { result += (c? ",D=" : "D=") + plask::str(*cache.D); c= true; }
    if (cache.dens) { result += (c? ",dens=" : "dens=") + plask::str(*cache.dens); c= true; }
    if (cache.Dso) { result += (c? ",Dso=" : "Dso=") + plask::str(*cache.Dso); c= true; }
    if (cache.e13) { result += (c? ",e13=" : "e13=") + plask::str(*cache.e13); c= true; }
    if (cache.e15) { result += (c? ",e15=" : "e15=") + plask::str(*cache.e15); c= true; }
    if (cache.e33) { result += (c? ",e33=" : "e33=") + plask::str(*cache.e33); c= true; }
    if (cache.EactA) { result += (c? ",EactA=" : "EactA=") + plask::str(*cache.EactA); c= true; }
    if (cache.EactD) { result += (c? ",EactD=" : "EactD=") + plask::str(*cache.EactD); c= true; }
    if (cache.Eg) { result += (c? ",Eg=" : "Eg=") + plask::str(*cache.Eg); c= true; }
    if (cache.eps) { result += (c? ",eps=" : "eps=") + plask::str(*cache.eps); c= true; }
    if (cache.lattC) { result += (c? ",lattC=" : "lattC=") + plask::str(*cache.lattC); c= true; }
    if (cache.Me) { result += (c? ",Me=" : "Me=") + plask::str(*cache.Me); c= true; }
    if (cache.Mh) { result += (c? ",Mh=" : "Mh=") + plask::str(*cache.Mh); c= true; }
    if (cache.Mhh) { result += (c? ",Mhh=" : "Mhh=") + plask::str(*cache.Mhh); c= true; }
    if (cache.Mlh) { result += (c? ",Mlh=" : "Mlh=") + plask::str(*cache.Mlh); c= true; }
    if (cache.mob) { result += (c? ",mob=" : "mob=") + plask::str(*cache.mob); c= true; }
    if (cache.mobe) { result += (c? ",mobe=" : "mobe=") + plask::str(*cache.mobe); c= true; }
    if (cache.mobh) { result += (c? ",mobh=" : "mobh=") + plask::str(*cache.mobh); c= true; }
    if (cache.Mso) { result += (c? ",Mso=" : "Mso=") + plask::str(*cache.Mso); c= true; }
    if (cache.Na) { result += (c? ",Na=" : "Na=") + plask::str(*cache.Na); c= true; }
    if (cache.Nd) { result += (c? ",Nd=" : "Nd=") + plask::str(*cache.Nd); c= true; }
    if (cache.Nf) { result += (c? ",Nf=" : "Nf=") + plask::str(*cache.Nf); c= true; }
    if (cache.Ni) { result += (c? ",Ni=" : "Ni=") + plask::str(*cache.Ni); c= true; }
    if (cache.nr) { result += (c? ",nr=" : "nr=") + plask::str(*cache.nr); c= true; }
    if (cache.Psp) { result += (c? ",Psp=" : "Psp=") + plask::str(*cache.Psp); c= true; }
    if (cache.taue) { result += (c? ",taue=" : "taue=") + plask::str(*cache.taue); c= true; }
    if (cache.tauh) { result += (c? ",tauh=" : "tauh=") + plask::str(*cache.tauh); c= true; }
    if (cache.thermk) { result += (c? ",thermk=" : "thermk=") + plask::str(*cache.thermk); c= true; }
    if (cache.VB) { result += (c? ",VB=" : "VB=") + plask::str(*cache.VB); c= true; }
    if (cache.y1) { result += (c? ",y1=" : "y1=") + plask::str(*cache.y1); c= true; }
    if (cache.y2) { result += (c? ",y2=" : "y2=") + plask::str(*cache.y2); c= true; }
    if (cache.y3) { result += (c? ",y3=" : "y3=") + plask::str(*cache.y3); c= true; }
    return "(" + result + ")";
}

double ConstMaterial::A(double) const { if (cache.A) return *cache.A; throwNotImplemented("A(double T)"); }

double ConstMaterial::absp(double, double) const { if (cache.absp) return *cache.absp; return 0.; }

double ConstMaterial::B(double) const { if (cache.B) return *cache.B; throwNotImplemented("B(double T)"); }

double ConstMaterial::C(double) const { if (cache.C) return *cache.C; throwNotImplemented("C(double T)"); }

double ConstMaterial::CB(double T, double e, char point) const {
    if (cache.CB) return *cache.CB;
    if (e == 0.)
        return VB(T, 0., point) + Eg(T, 0., point);
    else
        return max(VB(T, e, point, 'H'), VB(T, e, point, 'L')) + Eg(T, e, point);
}

double ConstMaterial::chi(double, double, char) const { if (cache.chi) return *cache.chi; throwNotImplemented("chi(double T, double e, char point)"); }

Tensor2<double> ConstMaterial::cond(double) const { if (cache.cond) return *cache.cond; throwNotImplemented("cond(double T)"); }

double ConstMaterial::D(double T) const {
    if (cache.D) return *cache.D;
    // Use Einstein relation here
    double mu;
    try { mu = mob(T).c00; }
    catch(plask::NotImplemented&) { if (cache.D) return *cache.D; throwNotImplemented("D(double T)"); }
    return mu * T * 8.6173423e-5;  // D = µ kB T / e
}

double ConstMaterial::dens(double) const { if (cache.dens) return *cache.dens; throwNotImplemented("dens(double T)"); }

double ConstMaterial::Dso(double, double) const { if (cache.Dso) return *cache.Dso; throwNotImplemented("Dso(double T, double e)"); }

double ConstMaterial::EactA(double) const { if (cache.EactA) return *cache.EactA; throwNotImplemented("EactA(double T)"); }
double ConstMaterial::EactD(double) const { if (cache.EactD) return *cache.EactD; throwNotImplemented("EactD(double T)"); }

double ConstMaterial::Eg(double, double, char) const { if (cache.Eg) return *cache.Eg; throwNotImplemented("Eg(double T, double e, char point)"); }

double ConstMaterial::eps(double) const { if (cache.eps) return *cache.eps; throwNotImplemented("eps(double T)"); }

double ConstMaterial::lattC(double, char) const { if (cache.lattC) return *cache.lattC; throwNotImplemented("lattC(double T, char x)"); }

Tensor2<double> ConstMaterial::Me(double, double, char) const { if (cache.Me) return *cache.Me; throwNotImplemented("Me(double T, double e, char point)"); }
Tensor2<double> ConstMaterial::Mh(double, double) const { if (cache.Mh) return *cache.Mh; throwNotImplemented("Mh(double T, double e)"); }
Tensor2<double> ConstMaterial::Mhh(double, double) const { if (cache.Mhh) return *cache.Mhh; throwNotImplemented("Mhh(double T, double e)"); }
Tensor2<double> ConstMaterial::Mlh(double, double) const { if (cache.Mlh) return *cache.Mlh; throwNotImplemented("Mlh(double T, double e)"); }

double ConstMaterial::y1() const { if (cache.y1) return *cache.y1; throwNotImplemented("y1()"); }
double ConstMaterial::y2() const { if (cache.y2) return *cache.y2; throwNotImplemented("y2()"); }
double ConstMaterial::y3() const { if (cache.y3) return *cache.y3; throwNotImplemented("y3()"); }

double ConstMaterial::ac(double) const { if (cache.ac) return *cache.ac; throwNotImplemented("ac(double T)"); }
double ConstMaterial::av(double) const { if (cache.av) return *cache.av; throwNotImplemented("av(double T)"); }
double ConstMaterial::b(double) const { if (cache.b) return *cache.b; throwNotImplemented("b(double T)"); }
double ConstMaterial::d(double) const { if (cache.d) return *cache.d; throwNotImplemented("d(double T)"); }
double ConstMaterial::c11(double) const { if (cache.c11) return *cache.c11; throwNotImplemented("c11(double T)"); }
double ConstMaterial::c12(double) const { if (cache.c12) return *cache.c12; throwNotImplemented("c12(double T)"); }
double ConstMaterial::c44(double) const { if (cache.c44) return *cache.c44; throwNotImplemented("c44(double T)"); }

Tensor2<double> ConstMaterial::mob(double) const { if (cache.mob) return *cache.mob; throwNotImplemented("mob(double T)"); }

double ConstMaterial::Mso(double, double) const { if (cache.Mso) return *cache.Mso; throwNotImplemented("Mso(double T, double e)"); }

double ConstMaterial::Nf(double) const { if (cache.Nf) return *cache.Nf; throwNotImplemented("Nf(double T)"); }

double ConstMaterial::Ni(double) const { if (cache.Ni) return *cache.Ni; throwNotImplemented("Ni(double T)"); }

double ConstMaterial::nr(double, double, double) const { if (cache.nr) return *cache.nr; throwNotImplemented("nr(double lam, double T, double n)"); }

dcomplex ConstMaterial::Nr(double lam, double T, double) const { return dcomplex(nr(lam,T), -7.95774715459e-09*absp(lam,T)*lam); }

Tensor3<dcomplex> ConstMaterial::NR(double lam, double T, double) const {
    return Nr(lam, T);
}

double ConstMaterial::cp(double) const { if (cache.cp) return *cache.cp; throwNotImplemented("cp(double T)"); }

Tensor2<double> ConstMaterial::thermk(double, double) const { if (cache.thermk) return *cache.thermk; throwNotImplemented("thermk(double T, double h)"); }

double ConstMaterial::VB(double, double, char, char) const { if (cache.VB) return *cache.VB; throwNotImplemented("VB(double T, double e, char point, char hole)"); }


Tensor2<double> ConstMaterial::mobe(double) const { if (cache.mobe) return *cache.mobe; throwNotImplemented("mobe(double T)"); }
Tensor2<double> ConstMaterial::mobh(double) const { if (cache.mobh) return *cache.mobh; throwNotImplemented("mobh(double T)"); }

double ConstMaterial::taue(double) const { if (cache.taue) return *cache.taue; throwNotImplemented("taue(double T)"); }

double ConstMaterial::tauh(double) const { if (cache.tauh) return *cache.tauh; throwNotImplemented("tauh(double T)"); }

double ConstMaterial::Ce(double) const { if (cache.Ce) return *cache.Ce; throwNotImplemented("Ce(double T)"); }

double ConstMaterial::Ch(double) const { if (cache.Ch) return *cache.Ch; throwNotImplemented("Ch(double T)"); }

double ConstMaterial::e13(double) const { if (cache.e13) return *cache.e13; throwNotImplemented("e13(double T)"); }

double ConstMaterial::e15(double) const { if (cache.e15) return *cache.e15; throwNotImplemented("e15(double T)"); }

double ConstMaterial::e33(double) const { if (cache.e33) return *cache.e33; throwNotImplemented("e33(double T)"); }

double ConstMaterial::c13(double) const { if (cache.c13) return *cache.c13; throwNotImplemented("c13(double T)"); }

double ConstMaterial::c33(double) const { if (cache.c33) return *cache.c33; throwNotImplemented("c33(double T)"); }

double ConstMaterial::Psp(double) const { if (cache.Psp) return *cache.Psp; throwNotImplemented("Psp(double T)"); }

double ConstMaterial::Na() const { if (cache.Na) return *cache.Na; throwNotImplemented("Na()"); }

double ConstMaterial::Nd() const { if (cache.Nd) return *cache.Nd; throwNotImplemented("Nd()"); }


}