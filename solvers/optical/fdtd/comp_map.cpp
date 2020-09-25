#ifndef COMP_MAP_H
#define COMP_MAP_H
#include <plask/plask.hpp>
#include <meep.hpp>

namespace plask { namespace solvers { namespace optical_fdtd {
    const std::map<std::string, meep::component> ComponentMap = 
    {
        {"ex", meep::Ex},
        {"ey", meep::Ey},
        {"er", meep::Er}, 	
        {"ep", meep::Ep}, 	
        {"ez", meep::Ez}, 	
        {"hx", meep::Hx}, 	
        {"hy", meep::Hy}, 	
        {"hr", meep::Hr}, 	
        {"hp", meep::Hp}, 	
        {"hz", meep::Hz}, 	
        {"dx", meep::Dx}, 	
        {"dy", meep::Dy}, 	
        {"dr", meep::Dr}, 	
        {"dp", meep::Dp}, 	
        {"dz", meep::Dz}, 	
        {"bx", meep::Bx}, 	
        {"by", meep::By}, 	
        {"br", meep::Br}, 	
        {"bp", meep::Bp}, 	
        {"bz", meep::Bz}, 	
        {"dielectric", meep::Dielectric}, 	
        {"permeability", meep::Permeability}
    };
}}};

#endif