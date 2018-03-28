#include "simple_optical_template.h"

namespace plask { namespace optical { namespace simple_optical_template {
    

template<typename Geometry2DType>
SimpleOpticalTemplate<Geometry2DType>::SimpleOpticalTemplate(const std::string& name):
    SolverOver<Geometry2DType>(name), 
    stripex(0)
{
    std::cout<<"Construktor " << std::endl;
}

template<typename Geometry2DType>
void SimpleOpticalTemplate<Geometry2DType>::loadConfiguration(XMLReader &reader, Manager &manager)
{
    // Load a configuration parameter from XML.
    while (reader.requireTagOrEnd()) {
        std::string param = reader.getNodeName();
        if (param == "mode") {
            stripex = reader.getAttribute<double>("vat", stripex);    
            stripex = reader.getAttribute<double>("lam0", stripex);
        } 
    }  
}

template<> std::string SimpleOpticalTemplate<Geometry2DCylindrical>::getClassName() const { return "simple_optical_template.SimpleOpticalCyl"; }
  
template struct PLASK_SOLVER_API SimpleOpticalTemplate<Geometry2DCylindrical>;

}}}