#include "semivectorial.h"

namespace plask { namespace optical { namespace semivectorial {
    

template<typename Geometry2DType>
SemiVectorial<Geometry2DType>::SemiVectorial(const std::string& name):
    SolverOver<Geometry2DType>(name), 
    stripex(0)
{}

template<typename Geometry2DType>
void SemiVectorial<Geometry2DType>::loadConfiguration(XMLReader &reader, Manager &manager)
{
    // Load a configuration parameter from XML.
    while (reader.requireTagOrEnd()) {
        std::string param = reader.getNodeName();
        if (param == "mode") {
        }
        else {     
            this->parseStandardConfiguration(reader, manager, "<geometry> or <root> or <mode>");}
        }    
}

template<typename Geometry2DType>
void SemiVectorial<Geometry2DType>::refractive_index(double x)
{
    onInitialize();
}

template<typename Geometry2DType>
void SemiVectorial<Geometry2DType>::onInitialize()
{
    if (!this->geometry) throw NoGeometryException(this->getId());
    shared_ptr<RectangularMesh<2>> mesh = makeGeometryGrid(this->geometry->getChild());
    shared_ptr<RectangularMesh<2>> midpoints = mesh->getMidpointsMesh();
    axis_vertical = mesh->vert();
    axis_horizontal = mesh->ee_x();
    axis_midpoints_vertical = midpoints->ee_y();
    axis_midpoints_horizontal = midpoints->ee_x();
    
    ybegin = 0;
    yend = mesh->axis1->size() + 1;
    for (double p: *axis_vertical)
    {
        edgeVertLayerPoint.push_back(p);  
    }
    double last_element = edgeVertLayerPoint.back();
    edgeVertLayerPoint.push_back(last_element+1e-3);    
    initializeRefractiveIndexVec();
}

template<typename Geometry2DType>
void SemiVectorial<Geometry2DType>::initializeRefractiveIndexVec()
{
    nrCache.clear();
    double T = 300; //temperature 300 K
    double wavelength = real(2e3*M_PI / k0);
    nrCache.push_back(this->geometry->getMaterial(vec(double(stripex),  0.0))->Nr(wavelength, T));
    for(double p: *axis_midpoints_vertical) { nrCache.push_back(this->geometry->getMaterial(vec(double(stripex),  p))->Nr(wavelength, T));}
    nrCache.push_back(this->geometry->getMaterial(vec(double(stripex),  edgeVertLayerPoint.back()))->Nr(wavelength, T));
}

template<> std::string SemiVectorial<Geometry2DCylindrical>::getClassName() const { return "optical.SemiVectorialCyl"; }
template<> std::string SemiVectorial<Geometry2DCartesian>::getClassName() const { return "optical.SemiVectorial2D"; }

template struct PLASK_SOLVER_API SemiVectorial<Geometry2DCylindrical>;
template struct PLASK_SOLVER_API SemiVectorial<Geometry2DCartesian>;


}}}