#ifndef PLASK__SOLVER_SIMPLE_OPTICAL_TEMPLATE
#define PLASK__SOLVER_SIMPLE_OPTICAL_TEMPLATE

#include <plask/plask.hpp>
#include "rootdigger.h"

namespace plask { namespace optical { namespace simple_optical_template {

template<typename Geometry2DType>
struct PLASK_SOLVER_API SimpleOpticalTemplate: public SolverOver<Geometry2DType>
{
        
    SimpleOpticalTemplate(const std::string& name="");
      
    virtual void loadConfiguration(XMLReader&, Manager&) override;
    
    virtual std::string getClassName() const override;
    
    //typename ProviderFor<LightMagnitude, Geometry2DType>::Delegate outLightMagnitude;
    
protected:
    
    double stripex;             ///< Position of the main stripe

};

}}}

#endif

