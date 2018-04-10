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
    axis_vertical = mesh->vert();
    axis_horizontal = mesh->ee_x();
    axis_midpoints_vertical = midpoints->ee_y();
    axis_midpoints_horizontal = midpoints->ee_x();
    ybegin = 0;
    yend = mesh->axis1->size() + 1;
    for (double p: *axis_vertical) edgeVertLayerPoint.push_back(p);  
    double last_element = edgeVertLayerPoint.back();
    edgeVertLayerPoint.push_back(last_element+1e-3);
    initializeRefractiveIndexVec();    
}

template<typename Geometry2DType>
void SimpleOpticalTemplate<Geometry2DType>::initializeRefractiveIndexVec()
{
    nrCache.clear();
    double T = 300; //temperature 300 K
    double wavelength = real(2e3*M_PI / k0);
    nrCache.push_back(this->geometry->getMaterial(vec(double(stripex),  0.0))->Nr(wavelength, T));
    for(double p: *axis_midpoints_vertical) { nrCache.push_back(this->geometry->getMaterial(vec(double(stripex),  p))->Nr(wavelength, T));}
    nrCache.push_back(this->geometry->getMaterial(vec(double(stripex),  edgeVertLayerPoint.back()))->Nr(wavelength, T));
}

template<typename Geometry2DType>
size_t SimpleOpticalTemplate<Geometry2DType>::findMode(double lambda)
{
    std::cout<<lambda<<std::endl;
    
    k0 = 2e3*M_PI/lambda;
    writelog(LOG_INFO, "Searching for the mode starting from wavelength = {0}", str(lambda));
    if (isnan(k0.real())) throw BadInput(this->getId(), "No reference wavelength `lam0` specified");
    onInitialize();
    Data2DLog<dcomplex,dcomplex> log_stripe(this->getId(), format(""), "", "");
        auto rootdigger = RootDigger::get(this, 
                        [&](const dcomplex& x ){
                        return this->computeTransferMatrix( (2e3*M_PI)/x, nrCache);},
                        log_stripe,
                        root);
    Mode mode(this);
    mode.lam = rootdigger->find((2e3*M_PI)/k0);
    return insertMode(mode);

}

template<typename Geometry2DType>
dcomplex SimpleOpticalTemplate<Geometry2DType>::computeTransferMatrix(const dcomplex& x, const std::vector<dcomplex>& NR)
{
    dcomplex w = 2e3*M_PI / x;
    setWavelength(w);
    Matrix phas_matrix(0,0,0,0);
    Matrix boundary_matrix(0,0,0,0);
    double d; //distance_between_layer
    Matrix transfer_matrix = Matrix::eye();
    dcomplex phas;
    vecE.clear();
    FieldZ field(0,1);
    vecE.push_back(field);
    for (size_t i = ybegin; i<yend-1; ++i)
    {
    if (i != ybegin || ybegin != 0) d = edgeVertLayerPoint[i] - edgeVertLayerPoint[i-1]; 
    else d = 0.;
    phas_matrix = Matrix(exp(I*NR[i]*x*d), 0, 0, exp(-I*NR[i]*x*d));
    boundary_matrix = Matrix( 0.5+0.5*(NR[i]/NR[i+1]), 0.5-0.5*(NR[i]/NR[i+1]),
                              0.5-0.5*(NR[i]/NR[i+1]), 0.5+0.5*(NR[i]/NR[i+1]) );
    transfer_matrix = (boundary_matrix*phas_matrix)*transfer_matrix;       
    FieldZ Ei = vecE[i]*(boundary_matrix*phas_matrix);
    vecE.push_back(Ei);} 

    return transfer_matrix.bb;
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