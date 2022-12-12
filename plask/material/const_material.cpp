/* 
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#include <boost/algorithm/string.hpp>

#include "const_material.hpp"
#include "db.hpp"
#include "../utils/lexical_cast.hpp"

namespace plask {

ConstMaterial::ConstMaterial(const std::string& full_name) {

    if (full_name[full_name.size()-1] != ']')
        throw MaterialParseException("{}: material with custom parameters must end with ']'", full_name);

    auto b = full_name.find('[');
    std::string basename = full_name.substr(0, b);
    boost::algorithm::trim_right(basename);
    if (!basename.empty()) base = MaterialsDB::getDefault().get(basename);

    std::string definition = full_name.substr(b+1, full_name.size()-b-2);

    for (auto item: boost::tokenizer<boost::char_separator<char>>(definition, boost::char_separator<char>(" ,"))) {
        std::string key, value;
        std::tie(key, value) = splitString2(item, '=');
        try {
            if (key.empty()) {
                continue;
            } else if (key == "lattC") {
                if (cache.lattC) throw MaterialParseException("{}: Repeated parameter 'lattC'", full_name);
                cache.lattC.reset(boost::lexical_cast<double>(value));
            } else if (key == "Eg") {
                if (cache.Eg) throw MaterialParseException("{}: Repeated parameter 'Eg'", full_name);
                cache.Eg.reset(boost::lexical_cast<double>(value));
            } else if (key == "CB") {
                if (cache.CB) throw MaterialParseException("{}: Repeated parameter 'CB'", full_name);
                cache.CB.reset(boost::lexical_cast<double>(value));
            } else if (key == "VB") {
                if (cache.VB) throw MaterialParseException("{}: Repeated parameter 'VB'", full_name);
                cache.VB.reset(boost::lexical_cast<double>(value));
            } else if (key == "Dso") {
                if (cache.Dso) throw MaterialParseException("{}: Repeated parameter 'Dso'", full_name);
                cache.Dso.reset(boost::lexical_cast<double>(value));
            } else if (key == "Mso") {
                if (cache.Mso) throw MaterialParseException("{}: Repeated parameter 'Mso'", full_name);
                cache.Mso.reset(boost::lexical_cast<double>(value));
            } else if (key == "Me") {
                if (cache.Me) throw MaterialParseException("{}: Repeated parameter 'Me'", full_name);
                cache.Me.reset(boost::lexical_cast<double>(value));
            } else if (key == "Mhh") {
                if (cache.Mhh) throw MaterialParseException("{}: Repeated parameter 'Mhh'", full_name);
                cache.Mhh.reset(boost::lexical_cast<double>(value));
            } else if (key == "Mlh") {
                if (cache.Mlh) throw MaterialParseException("{}: Repeated parameter 'Mlh'", full_name);
                cache.Mlh.reset(boost::lexical_cast<double>(value));
            } else if (key == "Mh") {
                if (cache.Mh) throw MaterialParseException("{}: Repeated parameter 'Mh'", full_name);
                cache.Mh.reset(boost::lexical_cast<double>(value));
            } else if (key == "y1") {
                if (cache.y1) throw MaterialParseException("{}: Repeated parameter 'y1'", full_name);
                cache.y1.reset(boost::lexical_cast<double>(value));
            } else if (key == "y2") {
                if (cache.y2) throw MaterialParseException("{}: Repeated parameter 'y2'", full_name);
                cache.y2.reset(boost::lexical_cast<double>(value));
            } else if (key == "y3") {
                if (cache.y3) throw MaterialParseException("{}: Repeated parameter 'y3'", full_name);
                cache.y3.reset(boost::lexical_cast<double>(value));
            } else if (key == "ac") {
                if (cache.ac) throw MaterialParseException("{}: Repeated parameter 'ac'", full_name);
                cache.ac.reset(boost::lexical_cast<double>(value));
            } else if (key == "av") {
                if (cache.av) throw MaterialParseException("{}: Repeated parameter 'av'", full_name);
                cache.av.reset(boost::lexical_cast<double>(value));
            } else if (key == "b") {
                if (cache.b) throw MaterialParseException("{}: Repeated parameter 'b'", full_name);
                cache.b.reset(boost::lexical_cast<double>(value));
            } else if (key == "d") {
                if (cache.d) throw MaterialParseException("{}: Repeated parameter 'd'", full_name);
                cache.d.reset(boost::lexical_cast<double>(value));
            } else if (key == "c11") {
                if (cache.c11) throw MaterialParseException("{}: Repeated parameter 'c11'", full_name);
                cache.c11.reset(boost::lexical_cast<double>(value));
            } else if (key == "c12") {
                if (cache.c12) throw MaterialParseException("{}: Repeated parameter 'c12'", full_name);
                cache.c12.reset(boost::lexical_cast<double>(value));
            } else if (key == "c44") {
                if (cache.c44) throw MaterialParseException("{}: Repeated parameter 'c44'", full_name);
                cache.c44.reset(boost::lexical_cast<double>(value));
            } else if (key == "eps") {
                if (cache.eps) throw MaterialParseException("{}: Repeated parameter 'eps'", full_name);
                cache.eps.reset(boost::lexical_cast<double>(value));
            } else if (key == "chi") {
                if (cache.chi) throw MaterialParseException("{}: Repeated parameter 'chi'", full_name);
                cache.chi.reset(boost::lexical_cast<double>(value));
            } else if (key == "Ni") {
                if (cache.Ni) throw MaterialParseException("{}: Repeated parameter 'Ni'", full_name);
                cache.Ni.reset(boost::lexical_cast<double>(value));
            } else if (key == "Nf") {
                if (cache.Nf) throw MaterialParseException("{}: Repeated parameter 'Nf'", full_name);
                cache.Nf.reset(boost::lexical_cast<double>(value));
            } else if (key == "EactD") {
                if (cache.EactD) throw MaterialParseException("{}: Repeated parameter 'EactD'", full_name);
                cache.EactD.reset(boost::lexical_cast<double>(value));
            } else if (key == "EactA") {
                if (cache.EactA) throw MaterialParseException("{}: Repeated parameter 'EactA'", full_name);
                cache.EactA.reset(boost::lexical_cast<double>(value));
            } else if (key == "mob") {
                if (cache.mob) throw MaterialParseException("{}: Repeated parameter 'mob'", full_name);
                cache.mob.reset(boost::lexical_cast<double>(value));
            } else if (key == "cond") {
                if (cache.cond) throw MaterialParseException("{}: Repeated parameter 'cond'", full_name);
                cache.cond.reset(boost::lexical_cast<double>(value));
            } else if (key == "A") {
                if (cache.A) throw MaterialParseException("{}: Repeated parameter 'A'", full_name);
                cache.A.reset(boost::lexical_cast<double>(value));
            } else if (key == "B") {
                if (cache.B) throw MaterialParseException("{}: Repeated parameter 'B'", full_name);
                cache.B.reset(boost::lexical_cast<double>(value));
            } else if (key == "C") {
                if (cache.C) throw MaterialParseException("{}: Repeated parameter 'C'", full_name);
                cache.C.reset(boost::lexical_cast<double>(value));
            } else if (key == "D") {
                if (cache.D) throw MaterialParseException("{}: Repeated parameter 'D'", full_name);
                cache.D.reset(boost::lexical_cast<double>(value));
            } else if (key == "thermk") {
                if (cache.thermk) throw MaterialParseException("{}: Repeated parameter 'thermk'", full_name);
                cache.thermk.reset(boost::lexical_cast<double>(value));
            } else if (key == "dens") {
                if (cache.dens) throw MaterialParseException("{}: Repeated parameter 'dens'", full_name);
                cache.dens.reset(boost::lexical_cast<double>(value));
            } else if (key == "cp") {
                if (cache.cp) throw MaterialParseException("{}: Repeated parameter 'cp'", full_name);
                cache.cp.reset(boost::lexical_cast<double>(value));
            } else if (key == "nr") {
                if (cache.nr) throw MaterialParseException("{}: Repeated parameter 'nr'", full_name);
                cache.nr.reset(boost::lexical_cast<double>(value));
            } else if (key == "absp") {
                if (cache.absp) throw MaterialParseException("{}: Repeated parameter 'absp'", full_name);
                cache.absp.reset(boost::lexical_cast<double>(value));
            } else if (key == "Nr") {
                if (cache.nr) throw MaterialParseException("{}: Repeated parameter 'Nr'", full_name);
                cache.Nr.reset(boost::lexical_cast<dcomplex>(value));
            } else if (key == "mobe") {
                if (cache.mobe) throw MaterialParseException("{}: Repeated parameter 'mobe'", full_name);
                cache.mobe.reset(boost::lexical_cast<double>(value));
            } else if (key == "mobh") {
                if (cache.mobh) throw MaterialParseException("{}: Repeated parameter 'mobh'", full_name);
                cache.mobh.reset(boost::lexical_cast<double>(value));
            } else if (key == "taue") {
                if (cache.taue) throw MaterialParseException("{}: Repeated parameter 'taue'", full_name);
                cache.taue.reset(boost::lexical_cast<double>(value));
            } else if (key == "tauh") {
                if (cache.tauh) throw MaterialParseException("{}: Repeated parameter 'tauh'", full_name);
                cache.tauh.reset(boost::lexical_cast<double>(value));
            } else if (key == "Ce") {
                if (cache.Ce) throw MaterialParseException("{}: Repeated parameter 'Ce'", full_name);
                cache.Ce.reset(boost::lexical_cast<double>(value));
            } else if (key == "Ch") {
                if (cache.Ch) throw MaterialParseException("{}: Repeated parameter 'Ch'", full_name);
                cache.Ch.reset(boost::lexical_cast<double>(value));
            } else if (key == "e13") {
                if (cache.e13) throw MaterialParseException("{}: Repeated parameter 'e13'", full_name);
                cache.e13.reset(boost::lexical_cast<double>(value));
            } else if (key == "e15") {
                if (cache.e15) throw MaterialParseException("{}: Repeated parameter 'e15'", full_name);
                cache.e15.reset(boost::lexical_cast<double>(value));
            } else if (key == "e33") {
                if (cache.e33) throw MaterialParseException("{}: Repeated parameter 'e33'", full_name);
                cache.e33.reset(boost::lexical_cast<double>(value));
            } else if (key == "c13") {
                if (cache.c13) throw MaterialParseException("{}: Repeated parameter 'c13'", full_name);
                cache.c13.reset(boost::lexical_cast<double>(value));
            } else if (key == "c33") {
                if (cache.c33) throw MaterialParseException("{}: Repeated parameter 'c33'", full_name);
                cache.c33.reset(boost::lexical_cast<double>(value));
            } else if (key == "Psp") {
                if (cache.Psp) throw MaterialParseException("{}: Repeated parameter 'Psp'", full_name);
                cache.Psp.reset(boost::lexical_cast<double>(value));
            } else if (key == "Na") {
                if (cache.Na) throw MaterialParseException("{}: Repeated parameter 'Na'", full_name);
                cache.Na.reset(boost::lexical_cast<double>(value));
            } else if (key == "Nd") {
                if (cache.Nd) throw MaterialParseException("{}: Repeated parameter 'Nd'", full_name);
                cache.Nd.reset(boost::lexical_cast<double>(value));
            } else throw MaterialParseException("{}: Bad material parameter '{}'", full_name, key);
        } catch (boost::bad_lexical_cast&) {
            throw MaterialParseException("{}: Bad material parameter value '{}={}'", full_name, key, value);
        }
    }

}

ConstMaterial::ConstMaterial(const shared_ptr<Material>& base, const std::map<std::string, double>& items):
    MaterialWithBase(base) {
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
        else if (item.first == "Nr") cache.Nr.reset(item.second);
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
    if (cache.A) { result += (c? " A=" : "A=") + plask::str(*cache.A); c= true; }
    if (cache.absp) { result += (c? " absp=" : "absp=") + plask::str(*cache.absp); c= true; }
    if (cache.ac) { result += (c? " ac=" : "ac=") + plask::str(*cache.ac); c= true; }
    if (cache.av) { result += (c? " av=" : "av=") + plask::str(*cache.av); c= true; }
    if (cache.b) { result += (c? " b=" : "b=") + plask::str(*cache.b); c= true; }
    if (cache.B) { result += (c? " B=" : "B=") + plask::str(*cache.B); c= true; }
    if (cache.C) { result += (c? " C=" : "C=") + plask::str(*cache.C); c= true; }
    if (cache.c11) { result += (c? " c11=" : "c11=") + plask::str(*cache.c11); c= true; }
    if (cache.c12) { result += (c? " c12=" : "c12=") + plask::str(*cache.c12); c= true; }
    if (cache.c13) { result += (c? " c13=" : "c13=") + plask::str(*cache.c13); c= true; }
    if (cache.c33) { result += (c? " c33=" : "c33=") + plask::str(*cache.c33); c= true; }
    if (cache.c44) { result += (c? " c44=" : "c44=") + plask::str(*cache.c44); c= true; }
    if (cache.CB) { result += (c? " CB=" : "CB=") + plask::str(*cache.CB); c= true; }
    if (cache.Ce) { result += (c? " Ce=" : "Ce=") + plask::str(*cache.Ce); c= true; }
    if (cache.Ch) { result += (c? " Ch=" : "Ch=") + plask::str(*cache.Ch); c= true; }
    if (cache.chi) { result += (c? " chi=" : "chi=") + plask::str(*cache.chi); c= true; }
    if (cache.cond) { result += (c? " cond=" : "cond=") + plask::str(*cache.cond); c= true; }
    if (cache.cp) { result += (c? " cp=" : "cp=") + plask::str(*cache.cp); c= true; }
    if (cache.d) { result += (c? " d=" : "d=") + plask::str(*cache.d); c= true; }
    if (cache.D) { result += (c? " D=" : "D=") + plask::str(*cache.D); c= true; }
    if (cache.dens) { result += (c? " dens=" : "dens=") + plask::str(*cache.dens); c= true; }
    if (cache.Dso) { result += (c? " Dso=" : "Dso=") + plask::str(*cache.Dso); c= true; }
    if (cache.e13) { result += (c? " e13=" : "e13=") + plask::str(*cache.e13); c= true; }
    if (cache.e15) { result += (c? " e15=" : "e15=") + plask::str(*cache.e15); c= true; }
    if (cache.e33) { result += (c? " e33=" : "e33=") + plask::str(*cache.e33); c= true; }
    if (cache.EactA) { result += (c? " EactA=" : "EactA=") + plask::str(*cache.EactA); c= true; }
    if (cache.EactD) { result += (c? " EactD=" : "EactD=") + plask::str(*cache.EactD); c= true; }
    if (cache.Eg) { result += (c? " Eg=" : "Eg=") + plask::str(*cache.Eg); c= true; }
    if (cache.eps) { result += (c? " eps=" : "eps=") + plask::str(*cache.eps); c= true; }
    if (cache.lattC) { result += (c? " lattC=" : "lattC=") + plask::str(*cache.lattC); c= true; }
    if (cache.Me) { result += (c? " Me=" : "Me=") + plask::str(*cache.Me); c= true; }
    if (cache.Mh) { result += (c? " Mh=" : "Mh=") + plask::str(*cache.Mh); c= true; }
    if (cache.Mhh) { result += (c? " Mhh=" : "Mhh=") + plask::str(*cache.Mhh); c= true; }
    if (cache.Mlh) { result += (c? " Mlh=" : "Mlh=") + plask::str(*cache.Mlh); c= true; }
    if (cache.mob) { result += (c? " mob=" : "mob=") + plask::str(*cache.mob); c= true; }
    if (cache.mobe) { result += (c? " mobe=" : "mobe=") + plask::str(*cache.mobe); c= true; }
    if (cache.mobh) { result += (c? " mobh=" : "mobh=") + plask::str(*cache.mobh); c= true; }
    if (cache.Mso) { result += (c? " Mso=" : "Mso=") + plask::str(*cache.Mso); c= true; }
    if (cache.Na) { result += (c? " Na=" : "Na=") + plask::str(*cache.Na); c= true; }
    if (cache.Nd) { result += (c? " Nd=" : "Nd=") + plask::str(*cache.Nd); c= true; }
    if (cache.Nf) { result += (c? " Nf=" : "Nf=") + plask::str(*cache.Nf); c= true; }
    if (cache.Ni) { result += (c? " Ni=" : "Ni=") + plask::str(*cache.Ni); c= true; }
    if (cache.nr) { result += (c? " nr=" : "nr=") + plask::str(*cache.nr); c= true; }
    if (cache.Nr) { result += (c? " Nr=" : "Nr=") + plask::str(*cache.Nr); c= true; }
    if (cache.Psp) { result += (c? " Psp=" : "Psp=") + plask::str(*cache.Psp); c= true; }
    if (cache.taue) { result += (c? " taue=" : "taue=") + plask::str(*cache.taue); c= true; }
    if (cache.tauh) { result += (c? " tauh=" : "tauh=") + plask::str(*cache.tauh); c= true; }
    if (cache.thermk) { result += (c? " thermk=" : "thermk=") + plask::str(*cache.thermk); c= true; }
    if (cache.VB) { result += (c? " VB=" : "VB=") + plask::str(*cache.VB); c= true; }
    if (cache.y1) { result += (c? " y1=" : "y1=") + plask::str(*cache.y1); c= true; }
    if (cache.y2) { result += (c? " y2=" : "y2=") + plask::str(*cache.y2); c= true; }
    if (cache.y3) { result += (c? " y3=" : "y3=") + plask::str(*cache.y3); c= true; }
    if (base)
        return base->str() + " [" + result + "]";
    else
        return "[" + result + "]";
}

double ConstMaterial::A(double T) const {
    if (cache.A) return *cache.A;
    else if (base) return base->A(T);
    else throwNotImplemented("A(double T)");
}

double ConstMaterial::absp(double, double) const { if (cache.absp) return *cache.absp; return 0.; }

double ConstMaterial::B(double T) const {
    if (cache.B) return *cache.B;
    else if (base) return base->B(T);
    else throwNotImplemented("A(double T)");
}

double ConstMaterial::C(double T) const {
    if (cache.C) return *cache.C;
    else if (base) return base->C(T);
    else throwNotImplemented("A(double T)");
}

double ConstMaterial::CB(double T, double e, char point) const {
    if (cache.CB) return *cache.CB;
    try {
        if (e == 0.)
            return VB(T, 0., point) + Eg(T, 0., point);
        else
            return max(VB(T, e, point, 'H'), VB(T, e, point, 'L')) + Eg(T, e, point);
    } catch (MaterialMethodNotImplemented) {
        if (base) return base->CB(T, e, point);
        else throwNotImplemented("CB(double T, double e, char point)");
    }
}

double ConstMaterial::chi(double T, double e, char point) const {
    if (cache.chi) return *cache.chi;
    else if (base) return base->chi(T, e, point);
    else throwNotImplemented("A(double T)");
}

Tensor2<double> ConstMaterial::cond(double T) const {
    if (cache.cond) return *cache.cond;
    else if (base) return base->cond(T);
    else throwNotImplemented("cond(double T)");
}

double ConstMaterial::D(double T) const {
    if (cache.D) return *cache.D;
    try {
        // Use Einstein relation here
        return mob(T).c00 * T * 8.6173423e-5;  // D = µ kB T / e
    } catch (MaterialMethodNotImplemented) {
        if (base) return base->D(T);
        else throwNotImplemented("D(double T)");
    }
}

double ConstMaterial::dens(double T) const {
    if (cache.dens) return *cache.dens;
    else if (base) return base->dens(T);
    else throwNotImplemented("A(double T)");
}

double ConstMaterial::Dso(double T, double e) const {
    if (cache.Dso) return *cache.Dso;
    else if (base) return base->Dso(T, e);
    else throwNotImplemented("A(double T)");
}

double ConstMaterial::EactA(double T) const {
    if (cache.EactA) return *cache.EactA;
    else if (base) return base->EactA(T);
    else throwNotImplemented("A(double T)");
}
double ConstMaterial::EactD(double T) const {
    if (cache.EactD) return *cache.EactD;
    else if (base) return base->EactD(T);
    else throwNotImplemented("A(double T)");
}

double ConstMaterial::Eg(double T, double e, char point) const {
    if (cache.Eg) return *cache.Eg;
    else if (base) return base->Eg(T, e, point);
    else throwNotImplemented("A(double T)");
}

double ConstMaterial::eps(double T) const {
    if (cache.eps) return *cache.eps;
    else if (base) return base->eps(T);
    else throwNotImplemented("A(double T)");
}

double ConstMaterial::lattC(double T, char x) const {
    if (cache.lattC) return *cache.lattC;
    else if (base) return base->lattC(T, x);
    else throwNotImplemented("A(double T)");
}

Tensor2<double> ConstMaterial::Me(double T, double e, char point) const {
    if (cache.Me) return *cache.Me;
    else if (base) return base->Me(T, e, point);
    else throwNotImplemented("A(double T)");
}
Tensor2<double> ConstMaterial::Mh(double T, double e) const {
    if (cache.Mh) return *cache.Mh;
    else if (base) return base->Mh(T, e);
    else throwNotImplemented("A(double T)");
}
Tensor2<double> ConstMaterial::Mhh(double T, double e) const {
    if (cache.Mhh) return *cache.Mhh;
    else if (base) return base->Mhh(T, e);
    else throwNotImplemented("A(double T)");
}
Tensor2<double> ConstMaterial::Mlh(double T, double e) const {
    if (cache.Mlh) return *cache.Mlh;
    else if (base) return base->Mlh(T, e);
    else throwNotImplemented("A(double T)");
}

double ConstMaterial::y1() const {
    if (cache.y1) return *cache.y1;
    else if (base) return base->y1();
    else throwNotImplemented("y1()");
}
double ConstMaterial::y2() const {
    if (cache.y2) return *cache.y2;
    else if (base) return base->y2();
    else throwNotImplemented("y2()");
}
double ConstMaterial::y3() const {
    if (cache.y3) return *cache.y3;
    else if (base) return base->y3();
    else throwNotImplemented("y3()");
}

double ConstMaterial::ac(double T) const {
    if (cache.ac) return *cache.ac;
    else if (base) return base->ac(T);
    else throwNotImplemented("A(double T)");
}
double ConstMaterial::av(double T) const {
    if (cache.av) return *cache.av;
    else if (base) return base->av(T);
    else throwNotImplemented("A(double T)");
}
double ConstMaterial::b(double T) const {
    if (cache.b) return *cache.b;
    else if (base) return base->b(T);
    else throwNotImplemented("A(double T)");
}
double ConstMaterial::d(double T) const {
    if (cache.d) return *cache.d;
    else if (base) return base->d(T);
    else throwNotImplemented("A(double T)");
}
double ConstMaterial::c11(double T) const {
    if (cache.c11) return *cache.c11;
    else if (base) return base->c11(T);
    else throwNotImplemented("A(double T)");
}
double ConstMaterial::c12(double T) const {
    if (cache.c12) return *cache.c12;
    else if (base) return base->c12(T);
    else throwNotImplemented("A(double T)");
}
double ConstMaterial::c44(double T) const {
    if (cache.c44) return *cache.c44;
    else if (base) return base->c44(T);
    else throwNotImplemented("A(double T)");
}

Tensor2<double> ConstMaterial::mob(double T) const {
    if (cache.mob) return *cache.mob;
    else if (base) return base->mob(T);
    else throwNotImplemented("A(double T)");
}

double ConstMaterial::Mso(double T, double e) const {
    if (cache.Mso) return *cache.Mso;
    else if (base) return base->Mso(T, e);
    else throwNotImplemented("A(double T)");
}

double ConstMaterial::Nf(double T) const {
    if (cache.Nf) return *cache.Nf;
    else if (base) return base->Nf(T);
    else throwNotImplemented("A(double T)");
}

double ConstMaterial::Ni(double T) const {
    if (cache.Ni) return *cache.Ni;
    else if (base) return base->Ni(T);
    else throwNotImplemented("A(double T)");
}

double ConstMaterial::nr(double lam, double T, double n) const {
    if (cache.nr) return *cache.nr;
    else if (base) return base->nr(lam, T, n);
    else throwNotImplemented("A(double T)");
}

dcomplex ConstMaterial::Nr(double lam, double T, double n) const {
    if (cache.Nr) return *cache.Nr;
    try {
        return dcomplex(nr(lam,T), -7.95774715459e-09*absp(lam,T)*lam);
    } catch (MaterialMethodNotImplemented) {
        if (base) return base->Nr(lam, T, n);
        else throwNotImplemented("Nr(double lam, double T, double n)");
    }
}

Tensor3<dcomplex> ConstMaterial::NR(double lam, double T, double n) const {
    try {
        return Nr(lam, T, n);
    } catch (MaterialMethodNotImplemented) {
        if (base) return base->NR(lam, T, n);
        else throwNotImplemented("NR(double lam, double T, double n)");
    }
}

double ConstMaterial::cp(double T) const {
    if (cache.cp) return *cache.cp;
    else if (base) return base->cp(T);
    else throwNotImplemented("A(double T)");
}

Tensor2<double> ConstMaterial::thermk(double T, double h) const {
    if (cache.thermk) return *cache.thermk;
    else if (base) return base->thermk(T, h);
    else throwNotImplemented("A(double T)");
}

double ConstMaterial::VB(double T, double e, char point, char hole) const {
    if (cache.VB) return *cache.VB;
    else if (base) return base->VB(T, e, point, hole);
    else throwNotImplemented("A(double T)");
}


Tensor2<double> ConstMaterial::mobe(double T) const {
    if (cache.mobe) return *cache.mobe;
    else if (base) return base->mobe(T);
    else throwNotImplemented("A(double T)");
}
Tensor2<double> ConstMaterial::mobh(double T) const {
    if (cache.mobh) return *cache.mobh;
    else if (base) return base->mobh(T);
    else throwNotImplemented("A(double T)");
}

double ConstMaterial::taue(double T) const {
    if (cache.taue) return *cache.taue;
    else if (base) return base->taue(T);
    else throwNotImplemented("A(double T)");
}

double ConstMaterial::tauh(double T) const {
    if (cache.tauh) return *cache.tauh;
    else if (base) return base->tauh(T);
    else throwNotImplemented("A(double T)");
}

double ConstMaterial::Ce(double T) const {
    if (cache.Ce) return *cache.Ce;
    else if (base) return base->Ce(T);
    else throwNotImplemented("A(double T)");
}

double ConstMaterial::Ch(double T) const {
    if (cache.Ch) return *cache.Ch;
    else if (base) return base->Ch(T);
    else throwNotImplemented("A(double T)");
}

double ConstMaterial::e13(double T) const {
    if (cache.e13) return *cache.e13;
    else if (base) return base->e13(T);
    else throwNotImplemented("A(double T)");
}

double ConstMaterial::e15(double T) const {
    if (cache.e15) return *cache.e15;
    else if (base) return base->e15(T);
    else throwNotImplemented("A(double T)");
}

double ConstMaterial::e33(double T) const {
    if (cache.e33) return *cache.e33;
    else if (base) return base->e33(T);
    else throwNotImplemented("A(double T)");
}

double ConstMaterial::c13(double T) const {
    if (cache.c13) return *cache.c13;
    else if (base) return base->c13(T);
    else throwNotImplemented("A(double T)");
}

double ConstMaterial::c33(double T) const {
    if (cache.c33) return *cache.c33;
    else if (base) return base->c33(T);
    else throwNotImplemented("A(double T)");
}

double ConstMaterial::Psp(double T) const {
    if (cache.Psp) return *cache.Psp;
    else if (base) return base->Psp(T);
    else throwNotImplemented("A(double T)");
}

double ConstMaterial::Na() const {
    if (cache.Na) return *cache.Na;
    else if (base) return base->Na();
    else throwNotImplemented("Na()");
}

double ConstMaterial::Nd() const {
    if (cache.Nd) return *cache.Nd;
    else if (base) return base->Nd();
    else throwNotImplemented("Nd()");
}


} // namespace plask
