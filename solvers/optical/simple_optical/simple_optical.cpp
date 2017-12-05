#include "simple_optical.h"

namespace plask { namespace optical { namespace simple_optical {

  

SimpleOptical::SimpleOptical(const std::string& name):plask::SolverOver<plask::Geometry2DCylindrical>(name)
{

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


void SimpleOptical::simpleVerticalSolver(double wave_length)
{
    t_bb = 0;
    setWavelength(wave_length);
    onInitialize();
    t_bb = compute_transfer_matrix(k0, refractive_index_vec);
    refractive_index_vec.clear();  
}

dcomplex SimpleOptical::get_T_bb()
{
  return t_bb;
}

dcomplex SimpleOptical::compute_transfer_matrix(const dcomplex& x, const std::vector<dcomplex> & NR)
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
    std::cout<<"i = "<<i<<std::endl;
    std::cout<<"x = " << x << std::endl;
    std::cout<<"d = " << d <<std::endl;
    std::cout<<"NR [i] " << NR[i] << std::endl;
    phas_matrix = Matrix(exp(I*NR[i]*x*d), 0, 0, exp(-I*NR[i]*x*d));
    std::cout<<"exp = " << exp(I*NR[i]*x*d) << std::endl;
    //phas = exp(-I*NR[i]*x*d);
    boundary_matrix = Matrix( 0.5+0.5*(NR[i]/NR[i+1]), 0.5-0.5*(NR[i]/NR[i+1]),
			      0.5-0.5*(NR[i]/NR[i+1]), 0.5+0.5*(NR[i]/NR[i+1]) );
    transfer_matrix = (boundary_matrix*phas_matrix)*transfer_matrix;
    std::cout<<"T_bb = "<< transfer_matrix.bb << std::endl;    
  }
  return transfer_matrix.bb;
}

  
}}}



