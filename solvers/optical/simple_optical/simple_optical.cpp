#include "simple_optical.h"

namespace plask { namespace optical { namespace simple_optical {
  
SimpleOptical::SimpleOptical(const std::string& name):plask::SolverOver<plask::Geometry2DCylindrical>(name)
  ,outLightMagnitude(this, &SimpleOptical::getLightMagnitude, &SimpleOptical::nmodes)
{
  stripe_root.method = RootDigger::ROOT_MULLER;
  stripe_root.tolx = 1.0e-6;
  stripe_root.tolf_min = 1.0e-7;
  stripe_root.tolf_max = 1.0e-5;
  stripe_root.maxiter = 500;
}

void SimpleOptical::loadConfiguration(XMLReader& reader, Manager& manager) {
    // Load a configuration parameter from XML.
    while (reader.requireTagOrEnd()) {
            parseStandardConfiguration(reader, manager, "<geometry>");
    }
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
    for (double p: *axis_vertical) edge_vert_layer_point.push_back(p);  
    initialize_refractive_index_vec();
}

void SimpleOptical::onInvalidate()
{
  z.clear();
  zfields.clear();
}

void SimpleOptical::initialize_refractive_index_vec()
{
  
  double T = 300; //temperature 300 K
  double w = real(2e3*M_PI / k0);
  refractive_index_vec.push_back(geometry->getMaterial(vec(0.0, -1e-3))->Nr(w, T));
  for(double p: *axis_midpoints_vertical) 
  {
    refractive_index_vec.push_back(geometry->getMaterial(vec(0.0,  p))->Nr(w, T));
  }

  double last_element = 0;
  last_element = edge_vert_layer_point.back();
  refractive_index_vec.push_back(geometry->getMaterial(vec(0.0,  last_element+1e-3))->Nr(w, T));
}



size_t SimpleOptical::findMode(double lambda, int m)
{
    k0 = 2e3*M_PI/lambda;
    writelog(LOG_INFO, "Searching for the mode starting from wavelength = {0}", str(lambda));
    if (isnan(k0.real())) throw BadInput(getId(), "No reference wavelength `lam0` specified");
    onInitialize();
    Data2DLog<dcomplex,dcomplex> log_stripe(getId(), format(""), "", "");
    auto rootdigger = RootDigger::get(this, 
				      [&](const dcomplex& x ){
					return this->computeTransferMatrix( (2e3*M_PI)/x, refractive_index_vec);	
				      },
				      log_stripe,
				      stripe_root);
    Mode mode(this, m);
    mode.lam = rootdigger->find((2e3*M_PI)/k0);
    std::cout<<"root wavelength: "<<mode.lam<<std::endl;
    refractive_index_vec.clear();
    return insertMode(mode);
}

dcomplex SimpleOptical::computeTransferMatrix(const dcomplex& x, const std::vector<dcomplex> & NR)
{
  double w = real(2e3*M_PI / x);
  setWavelength(w);
  onInitialize();
  Matrix phas_matrix;
  Matrix boundary_matrix;
  double d; //distance_between_layer
  transfer_matrix = Matrix::eye();
  dcomplex phas;
  
  FieldZ field(0,1);
  vecE.push_back(field);
  
  for (size_t i = ybegin; i<yend-1; ++i)
  {
    if (i != ybegin || ybegin != 0) d = edge_vert_layer_point[i] - edge_vert_layer_point[i-1]; 
    else d = 0.;
    
    phas_matrix = Matrix(exp(I*NR[i]*x*d), 0, 0, exp(-I*NR[i]*x*d));
    boundary_matrix = Matrix( 0.5+0.5*(NR[i]/NR[i+1]), 0.5-0.5*(NR[i]/NR[i+1]),
                              0.5-0.5*(NR[i]/NR[i+1]), 0.5+0.5*(NR[i]/NR[i+1]) );
    dcomplex F = transfer_matrix.fb, B = transfer_matrix.bb;
    transfer_matrix = (boundary_matrix*phas_matrix)*transfer_matrix;    
    
    FieldZ Ei = vecE[i]*(boundary_matrix*phas_matrix);
    vecE.push_back(Ei);    
  }
  
  
  refractive_index_vec.clear();
  return transfer_matrix.bb;
}



std::vector<dcomplex> SimpleOptical::getNrCache()
{
  return nrCache;
}
const DataVector<double> SimpleOptical::getLightMagnitude(int num, const shared_ptr<const MeshD<2>>& dst_mesh, InterpolationMethod)
{
  setWavelength(real(modes[num].lam));
  
  onInitialize();
  vecE.clear();
  computeTransferMatrix(k0, refractive_index_vec);
  
  std::vector<double> arrayZ;  
  for (auto v: *dst_mesh) {
    double z = v.c1;
    arrayZ.push_back(z);
  }
 
  DataVector<double> results(arrayZ.size());
  std::vector<dcomplex> NR;
  std::vector<double> verticalEdgeVec;
  std::vector<double> hi;
  std::vector<dcomplex> B;
  std::vector<dcomplex> F;
  double T = 300; //temperature 300 K
  double w = real(2e3*M_PI / x);
  
 
  for(auto p: arrayZ) // TO DO maybe use dst mesh direct 
  {
    NR.push_back(geometry->getMaterial(vec(0.0,  p))->Nr(w, T));      
  }
   
  for (double p_edge: *axis_vertical) verticalEdgeVec.push_back(p_edge); 
  
  std::cout<<"edge size = " << verticalEdgeVec.size()<<std::endl;
  for (size_t i = 0; i < verticalEdgeVec.size()-1; ++i)
  {
    for (double p: arrayZ) 
    {

      if (verticalEdgeVec[i] <= p and verticalEdgeVec[i+1] > p)
      {
	std::cout<<p<<std::endl;
        z.push_back(p);
        hi.push_back(p - verticalEdgeVec[i]);        
        B.push_back(vecE[i+1].B);
        //std::cout<<"add B = "<<vecE[i+1].B<<std::endl;
        F.push_back(vecE[i+1].F);        
      }
    }
   }
  
   zfields.clear();
  
   dcomplex Ez;
   std::cout<<"hi size = "<<hi.size()-1 << std::endl;
   for (size_t i = 0; i < hi.size(); ++i)
   {
      std::cout<<"hi = "<<hi[i]<<" F = "<<F[i]<<" B = "<<B[i]<<std::endl;
      Ez = F[i]*exp(-I*NR[i]*k0*hi[i]) + B[i]*exp(I*NR[i]*k0*hi[i]); 
      zfields.push_back(Ez);
      results[i] = real(Ez + conj(Ez));
   }
  
  z = arrayZ;
  
  return results;
}




}}}






