#include "simple_optical_template.h"

namespace plask { namespace optical { namespace simple_optical_template {
    

template<typename Geometry2DType>
SimpleOpticalTemplate<Geometry2DType>::SimpleOpticalTemplate(const std::string& name):
    SolverOver<Geometry2DType>(name), 
    stripex(0),
    outLightMagnitude(this, &SimpleOpticalTemplate<Geometry2DType>::getLightMagnitude, &SimpleOpticalTemplate<Geometry2DType>::nmodes)
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
        else {     
            this->parseStandardConfiguration(reader, manager, "<geometry> or <root> or <mode>");}
        }    
}

template<typename Geometry2DType>
void SimpleOpticalTemplate<Geometry2DType>::onInitialize()
{
    if (!this->geometry) throw NoGeometryException(this->getId());
    shared_ptr<RectangularMesh<2>> mesh = makeGeometryGrid(this->geometry->getChild());
    shared_ptr<RectangularMesh<2>> midpoints = mesh->getMidpointsMesh();
    for (auto p : *midpoints)
    {
        std::cout<<p<<std::endl;
    }

}

template<typename Geometry2DType>
void SimpleOpticalTemplate<Geometry2DType>::findMode(double lambda)
{
    std::cout<<lambda<<std::endl;
    onInitialize();
}

template<typename Geometry2DType>
const DataVector<double> SimpleOpticalTemplate<Geometry2DType>::getLightMagnitude(int num, const shared_ptr<const MeshD<2>>& dst_mesh, InterpolationMethod)
{
    
}

template<typename Geometry2DType> void SimpleOpticalTemplate<Geometry2DType>::onInvalidate() {
    nrCache.clear();
}

template<> std::string SimpleOpticalTemplate<Geometry2DCylindrical>::getClassName() const { return "optical.SimpleOpticalCyl2D"; }
template<> std::string SimpleOpticalTemplate<Geometry2DCartesian>::getClassName() const { return "optical.SimpleOpticalCar2D"; }

template struct PLASK_SOLVER_API SimpleOpticalTemplate<Geometry2DCylindrical>;
template struct PLASK_SOLVER_API SimpleOpticalTemplate<Geometry2DCartesian>;

}}}