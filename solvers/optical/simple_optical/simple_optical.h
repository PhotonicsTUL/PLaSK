#ifndef PLASK__SOLVER_SIMPLE_OPTICAL
#define PLASK__SOLVER_SIMPLE_OPTICAL

#include <plask/plask.hpp>

namespace plask { namespace solvers { namespace simple_optical {

/**
 * This is Doxygen documentation of your solver.
 * Write a brief description of it.
 */
struct PLASK_SOLVER_API SimpleOptical: public SolverOver<Geometry2DCylindrical> {
  

   SimpleOptical(const std::string& name="SimpleOptical");
   
   void loadConfiguration(XMLReader& reader, Manager& manager);

   virtual std::string getClassName() const { return "SimpleOptical"; }
   
   virtual void onInitialize() {
      if (!geometry) throw NoGeometryException(getId());
   }
   
   //void setSimpleMesh() {
   //     writelog(LOG_INFO, "Creating simple mesh");
   //     setMesh(plask::make_shared<RectangularMesh2DSimpleGenerator>());
   // }

   void say_hello();
   
private:
   plask::DataVector<double> boundary_layer;
   std::string axis_name;

};
  
}}} // namespace

#endif