#include "simple_optical.h"

namespace plask { namespace optical { namespace simple_optical {

  

SimpleOptical::SimpleOptical(const std::string& name):plask::SolverOver<plask::Geometry2DCylindrical>(name)
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
    
    Data2DLog<dcomplex,dcomplex> log_stripe(getId(), format("stripe[{0}]", ybegin), "neff", "det");
     
 
 /*   auto rootdigger = RootDigger::get(this, [&](const dcomplex& x){return this->compute_transfer_matrix(k0, refractive_index_vec);}, log_stripe, stripe_root);
  
    if (vneff == 0.) {
            dcomplex maxn = *std::max_element(refractive_index_vec.begin(), refractive_index_vec.end(),
                                              [](const dcomplex& a, const dcomplex& b){return real(a) < real(b);} );
            vneff = 0.999 * real(maxn);
        }
    vneff = rootdigger->find(vneff);
    std::cout<<vneff<<std::endl;
   */ 
    refractive_index_vec.clear();  
}

void SimpleOptical::compute_electric_field_distribution(double wave_length)
{
  setWavelength(wave_length);
  onInitialize();
  eField = compute_eField(k0, refractive_index_vec);
  refractive_index_vec.clear();
}

dcomplex SimpleOptical::get_T_bb()
{
  return t_bb;
}

std::vector<dcomplex> SimpleOptical::get_eField()
{
  return eField;
}

std::vector<dcomplex> SimpleOptical::get_bField()
{
  return bField;
}

std::vector<double> SimpleOptical::get_z()
{
  return z;
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
    phas_matrix = Matrix(exp(I*NR[i]*x*d), 0, 0, exp(-I*NR[i]*x*d));
    boundary_matrix = Matrix( 0.5+0.5*(NR[i]/NR[i+1]), 0.5-0.5*(NR[i]/NR[i+1]),
			      0.5-0.5*(NR[i]/NR[i+1]), 0.5+0.5*(NR[i]/NR[i+1]) );
    
    dcomplex F = transfer_matrix.fb, B = transfer_matrix.bb;
    transfer_matrix = (boundary_matrix*phas_matrix)*transfer_matrix;    
  }
  return transfer_matrix.bb;
}

std::vector<dcomplex> SimpleOptical::compute_eField(const dcomplex& x, const std::vector<dcomplex> & NR)
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
    eField.push_back(F);
    z.push_back(edge_vert_layer_point[i]);
    transfer_matrix = (boundary_matrix*phas_matrix)*transfer_matrix;    
  }
  
  return eField;
}
  
}}}



