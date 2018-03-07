#include "simple_optical.h"

namespace plask { namespace optical { namespace simple_optical {
  
SimpleOptical::SimpleOptical(const std::string& name):plask::SolverOver<plask::Geometry2DCylindrical>(name)
    ,outLightMagnitude(this, &SimpleOptical::getLightMagnitude, &SimpleOptical::nmodes)
    ,outRefractiveIndex(this, &SimpleOptical::getRefractiveIndex)
    ,stripex(0)
{}

void SimpleOptical::loadConfiguration(XMLReader& reader, Manager& manager) {
    // Load a configuration parameter from XML.
    while (reader.requireTagOrEnd()) {
        std::string param = reader.getNodeName();
        if (param == "mode") {
            stripex = reader.getAttribute<double>("vat", stripex);    
            stripex = reader.getAttribute<double>("lam0", stripex);
        } else if (param == "root") {
            RootDigger::readRootDiggerConfig(reader, stripe_root); }
        else {     
            parseStandardConfiguration(reader, manager, "<geometry> or <root> or <mode>");}
    }    
}

void SimpleOptical::onInvalidate()
{
    if (!modes.empty()) {
        writelog(LOG_DETAIL, "Clearing computed modes");
        nrCache.clear();
        modes.clear();
        outLightMagnitude.fireChanged();
        outRefractiveIndex.fireChanged();}
}

void SimpleOptical::onInitialize()
{

    if (!geometry) throw NoGeometryException(getId());
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


void SimpleOptical::initializeRefractiveIndexVec()
{
    nrCache.clear();
    double T = 300; //temperature 300 K
    double w = real(2e3*M_PI / k0);
    nrCache.push_back(geometry->getMaterial(vec(double(stripex),  0.0))->Nr(w, T));
    for(double p: *axis_midpoints_vertical) {
    nrCache.push_back(geometry->getMaterial(vec(double(stripex),  p))->Nr(w, T));}
    nrCache.push_back(geometry->getMaterial(vec(double(stripex),  edgeVertLayerPoint.back()))->Nr(w, T));
}

size_t SimpleOptical::findMode(double lambda, int m)
{
    k0 = 2e3*M_PI/lambda;
    writelog(LOG_INFO, "Searching for the mode starting from wavelength = {0}", str(lambda));
    if (isnan(k0.real())) throw BadInput(getId(), "No reference wavelength `lam0` specified");
    onInitialize();
    initializeRefractiveIndexVec();
    Data2DLog<dcomplex,dcomplex> log_stripe(getId(), format(""), "", "");
    auto rootdigger = RootDigger::get(this, 
				      [&](const dcomplex& x ){
					return this->computeTransferMatrix( (2e3*M_PI)/x, nrCache);	
				      },
				      log_stripe,
				      stripe_root);
    Mode mode(this, m);
    mode.lam = rootdigger->find((2e3*M_PI)/k0);
    return insertMode(mode);
}

dcomplex SimpleOptical::computeTransferMatrix(const dcomplex& x, const std::vector<dcomplex> & NR)
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

void SimpleOptical::updateCache()
{
    bool fresh = initCalculation();

    if (fresh) {
        // we need to update something
        onInitialize();}
}

dcomplex SimpleOptical::getVertDeterminant(dcomplex wavelength)
{
    setWavelength(wavelength);
    updateCache();
    initializeRefractiveIndexVec();
    return computeTransferMatrix(k0, nrCache);
}

const LazyData<Tensor3<dcomplex>> SimpleOptical::getRefractiveIndex(const shared_ptr<const MeshD<2>> &dst_mesh, InterpolationMethod)
{
  this->writelog(LOG_DEBUG, "Getting refractive indices");
  dcomplex lam0 = 2e3*M_PI / k0;
  InterpolationFlags flags(geometry);
  return LazyData<Tensor3<dcomplex>>(dst_mesh->size(),
        [this, dst_mesh, flags, lam0](size_t j) -> Tensor3<dcomplex> {
            auto point = flags.wrap(dst_mesh->at(j));
	    return geometry->getMaterial(vec(double(stripex), dst_mesh->at(j)[1]))->Nr(real(lam0), 300) ;
        }
    );
}

const DataVector<double> SimpleOptical::getLightMagnitude(int num, const shared_ptr<const MeshD<2>>& dst_mesh, InterpolationMethod)
{
    std::cout<<"Stripex = " << stripex << std::endl;
    setWavelength((modes[num].lam)); 
    vecE.clear();
    k0 = 2e3*M_PI / modes[num].lam;
    
    computeTransferMatrix(k0, nrCache);
    std::vector<double> arrayZ;  
    for (auto v: *dst_mesh) {double z = v.c1;
                             arrayZ.push_back(z);}
 
    DataVector<double> results(arrayZ.size());
    std::vector<dcomplex> NR;
    std::vector<double> verticalEdgeVec;
    std::vector<double> hi;
    std::vector<dcomplex> B;
    std::vector<dcomplex> F;
    double T = 300; //temperature 300 K
    double w = real(2e3*M_PI / k0);
  
    for(auto p: arrayZ) NR.push_back(geometry->getMaterial(vec(double(stripex),  p))->Nr(w, T));
    //MD: dlaczego nie korzysta Pan z `nrCache`?
     
    
    for (double p_edge: *axis_vertical) verticalEdgeVec.push_back(p_edge);
    
    for (double p : arrayZ) 
      if (p<0){hi.push_back(-p);
               B.push_back(vecE[0].B);
               F.push_back(vecE[0].F);}
    
    for (size_t i = 0; i < verticalEdgeVec.size()-1; ++i)
    {
     for (double p: arrayZ) 
     {
      if (verticalEdgeVec[i] <= p and verticalEdgeVec[i+1] > p)
      {hi.push_back(p - verticalEdgeVec[i]);        
       B.push_back(vecE[i+1].B);
       F.push_back(vecE[i+1].F);}
    }}
    
    for(double p: arrayZ) // propagation wave after escape from structure 
    {
        if (p > verticalEdgeVec.back())
        {   
        hi.push_back(p-verticalEdgeVec.back());    
        B.push_back(vecE.back().B);
        F.push_back(vecE.back().F);
    }} 
    
    dcomplex Ez;
    for (size_t i = 0; i < hi.size(); ++i)
    {
       Ez = F[i]*exp(-I*NR[i]*k0*hi[i]) + B[i]*exp(I*NR[i]*k0*hi[i]); 
       results[i] = real(Ez*conj(Ez));
    }   
    return results;
}




}}}






