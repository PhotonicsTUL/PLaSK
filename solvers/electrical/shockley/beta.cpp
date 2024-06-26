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
#include "beta.hpp"

namespace plask { namespace electrical { namespace shockley {

template<typename Geometry2DType>
BetaSolver<Geometry2DType>::BetaSolver(const std::string& name) : BaseClass(name)
{
    js.assign(1, 1.);
    beta.assign(1, NAN);
}

template<typename Geometry2DType>
void BetaSolver<Geometry2DType>::loadConfiguration(XMLReader &source, Manager &manager)
{
    while (source.requireTagOrEnd())
    {
        if (source.getNodeName() == "junction") {
            js[0] = source.getAttribute<double>("js", js[0]);
            beta[0] = source.getAttribute<double>("beta", beta[0]);
            auto condjunc = source.getAttribute<double>("pnjcond");
            if (condjunc) {
                this->writelog(LOG_WARNING, "'pnjcond' attribute is obselete; use <loop start-cond=>");
                this->setCondJunc(*condjunc);
            }
            // if (source.hasAttribute("wavelength") || source.hasAttribute("heat"))
            //     throw XMLException(reader, "heat computation by wavelegth is no onger supporte");
            for (auto attr: source.getAttributes()) {
                if (attr.first == "beta" || attr.first == "js" || attr.first == "pnjcond" || attr.first == "wavelength" || attr.first == "heat") continue;
                if (attr.first.substr(0,4) == "beta") {
                    size_t no;
                    try { no = boost::lexical_cast<size_t>(attr.first.substr(4)); }
                    catch (boost::bad_lexical_cast&) { throw XMLUnexpectedAttrException(source, attr.first); }
                    setBeta(no, source.requireAttribute<double>(attr.first));
                }
                else if (attr.first.substr(0,2) == "js") {
                    size_t no;
                    try { no = boost::lexical_cast<size_t>(attr.first.substr(2)); }
                    catch (boost::bad_lexical_cast&) { throw XMLUnexpectedAttrException(source, attr.first); }
                    setJs(no, source.requireAttribute<double>(attr.first));
                }
                else
                    throw XMLUnexpectedAttrException(source, attr.first);
            }
            source.requireTagEnd();
        }
        else {
            this->parseConfiguration(source, manager);
        }
    }
}


template<typename Geometry2DType>
BetaSolver<Geometry2DType>::~BetaSolver() {
}

template<> std::string BetaSolver<Geometry2DCartesian>::getClassName() const { return "electrical.Shockley2D"; }
template<> std::string BetaSolver<Geometry2DCylindrical>::getClassName() const { return "electrical.ShockleyCyl"; }
template<> std::string BetaSolver<Geometry3D>::getClassName() const { return "electrical.Shockley3D"; }

template struct PLASK_SOLVER_API BetaSolver<Geometry2DCartesian>;
template struct PLASK_SOLVER_API BetaSolver<Geometry2DCylindrical>;
template struct PLASK_SOLVER_API BetaSolver<Geometry3D>;

}}} // namespaces
