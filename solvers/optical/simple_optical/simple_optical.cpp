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

void SimpleOptical::stageOne()
{
  
}

void SimpleOptical::simpleVerticalSolver(double wave_length)
{
    t_bb = 0;
    setWavelength(wave_length);
    onInitialize();
    t_bb = compute_transfer_matrix(k0, refractive_index_vec);
    //refractive_index_vec.clear();
}

dcomplex SimpleOptical::get_T_bb()
{
  return t_bb;
}

dcomplex SimpleOptical::findRoot(double k0)
{
    onInitialize();
    Data2DLog<dcomplex,dcomplex> log_stripe(getId(), format(""), "", "");
    auto rootdigger = RootDigger::get(this, 
				      [&](const dcomplex& x ){
					return this->compute_transfer_matrix( (2e3*M_PI)/x, refractive_index_vec);	
				      },
				      log_stripe,
				      stripe_root);
    vneff = rootdigger->find((2e3*M_PI)/k0);
    std::cout<<"root wavelength: "<<vneff<<std::endl;
    refractive_index_vec.clear();
    return vneff;
}

dcomplex SimpleOptical::compute_transfer_matrix(const dcomplex& x, const std::vector<dcomplex> & NR)
{
  double w = real(2e3*M_PI / x);
  setWavelength(w);
  onInitialize();
  Matrix phas_matrix;
  Matrix boundary_matrix;
  double d; //distance_between_layer
  transfer_matrix = Matrix::eye();
  dcomplex phas;

  for (size_t i = ybegin; i<yend-1; ++i)
  {
    if (i != ybegin || ybegin != 0) d = edge_vert_layer_point[i] - edge_vert_layer_point[i-1]; 
    else d = 0.;
    phas_matrix = Matrix(exp(I*NR[i]*x*d), 0, 0, exp(-I*NR[i]*x*d));
    boundary_matrix = Matrix( 0.5+0.5*(NR[i]/NR[i+1]), 0.5-0.5*(NR[i]/NR[i+1]),
                              0.5-0.5*(NR[i]/NR[i+1]), 0.5+0.5*(NR[i]/NR[i+1]) );
    dcomplex F = transfer_matrix.fb, B = transfer_matrix.bb;
    transfer_matrix = (boundary_matrix*phas_matrix)*transfer_matrix;    
  }
  refractive_index_vec.clear();
  std::cout<<"x = " << x << std::endl;
  std::cout<<"T bb = " << transfer_matrix.bb<<std::endl;
  return transfer_matrix.bb;
}

void SimpleOptical::computeField(double wavelength)
{
  setWavelength(wavelength);
  onInitialize();
  computeEz(k0, refractive_index_vec);
 
}

std::vector<double> SimpleOptical::getZ()
{
  return z;
}

std::vector<dcomplex> SimpleOptical::getEz()
{
  return zfields;
}

const LazyData<double> SimpleOptical::getLightMagnitude(int num, const shared_ptr<const MeshD<2>>& dst_mesh, InterpolationMethod)
{
}

std::vector<dcomplex> SimpleOptical::computeEz(const dcomplex& x, const std::vector<dcomplex> & NR)
{
  Matrix phas_matrix;
  Matrix boundary_matrix;
  double d; //distance_between_layer  
  transfer_matrix = Matrix::eye();
  dcomplex phas;
  for (size_t i = ybegin; i<yend-1; ++i)
  {
    if (i != ybegin || ybegin != 0) d = edge_vert_layer_point[i] - edge_vert_layer_point[i-1]; 
    else d = 0.;
    phas_matrix = Matrix(exp(I*NR[i]*x*d), 0, 0, exp(-I*NR[i]*x*d));
    boundary_matrix = Matrix( 0.5+0.5*(NR[i]/NR[i+1]), 0.5-0.5*(NR[i]/NR[i+1]),
                              0.5-0.5*(NR[i]/NR[i+1]), 0.5+0.5*(NR[i]/NR[i+1]) );
    dcomplex F = transfer_matrix.fb; 
    dcomplex B = transfer_matrix.bb;
    zfields.push_back(F+B);
    transfer_matrix = (boundary_matrix*phas_matrix)*transfer_matrix;        
  }  
  
  z.push_back(0);
  for (double p: *axis_midpoints_vertical)
  {
    z.push_back(p);
  }
  
  return zfields;
}


}}}






