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

void SimpleOptical::say_hello()
{
    shared_ptr<RectangularMesh<2>> mesh = makeGeometryGrid(this->geometry->getChild());
    shared_ptr<RectangularMesh<2>> points = mesh->getMidpointsMesh();

    std::cout<<"Hello world!!!!!!!!2222222222222 "<<std::endl;

    std::cout << "mesh: "; for (double p: *mesh->vert()) std::cout << p << " "; std::cout << std::endl;
    std::cout<< "Midpoint Mesh vertical: "; for (double p: *points->vert()) std::cout<<p<<" "; std::cout<< std::endl;
    std::cout<< "Midpoint Mesh horizontal "; for(double p: *points->ee_x()) std::cout<<p<<" "; std::cout<< std::endl;
    std::cout << "material: "; for (double p: *points->vert()) std::cout << geometry->getMaterial(vec(0.0, p))->name() << " "; std::cout << std::endl;
    std::cout << "nr: "; for (double p: *points->vert()) std::cout << str(geometry->getMaterial(vec(0.0, p))->Nr(1300, 300)) << " "; std::cout << std::endl;
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
    yend = axis_vertical->size();
    setWavelength(800);
    initialize_refractive_index_vec();
    std::cout<<"Wavelength: "<<getWavelength()<<std::endl;
}

void SimpleOptical::initialize_refractive_index_vec()
{
  double T = 300; //temperature 300 K
  for(double p: *axis_midpoints_vertical) 
  {
    refractive_index_vec.push_back((geometry->getMaterial(vec(0.5,  p))->Nr(getWavelength(), T)));
  }
  
  for (const dcomplex& i : refractive_index_vec)
  {
     std::cout<<i<<std::endl;
  }
}

void SimpleOptical::simpleVerticalSolver()
{
   
    onInitialize();
    double c = 3e8;
    double freq = c/getWavelength();
     
    comput_T_bb(freq, refractive_index_vec);
    std::cout<< comput_T_bb(freq, refractive_index_vec) << std::endl;  
  
}


// x - frequency
dcomplex SimpleOptical::comput_T_bb(const dcomplex& x, const std::vector< dcomplex >& NR)
{
  
    std::vector<dcomplex> ky(yend);
    for (size_t i = ybegin; i < yend; ++i) {
        ky[i] = k0 * sqrt(NR[i]*NR[i] - x*x);
        if (imag(ky[i]) > 0.) ky[i] = -ky[i];
    }
    
         
    std::cout<<"ky:"<<std::endl;
    for (const dcomplex& i : ky) std::cout<<i<<std::endl;
    
    std::vector<double> d;
    for (double p: *axis_vertical) d.push_back(p);
    
    
    Matrix T = Matrix::eye();
    dcomplex h_i;
    for (size_t i = ybegin; i < yend-1; ++i) {
	 std::cout<<"i = " << i << std::endl;
         std::cout<<"h_i = " << d[i+1]-d[i] << std::endl;
	 h_i = d[i+1]-d[i];
	 dcomplex phas = exp(- I * ky[i] * h_i*1e-6);
	 std::cout<<"phas = " << phas << std::endl;
	 std::cout<<"ky = " << ky[i] << std::endl;
	 std::cout<<"-I*ky*h_i = " << -I*ky[i]*h_i*1e-6 << std::endl;
	 //Transfer through boundary
         dcomplex f = (polarization==TM)? (NR[i+1]/NR[i]) : 1.;
         dcomplex n = 0.5 * ky[i]/ky[i+1] * f*f;
         Matrix T1 = Matrix( (0.5+n), (0.5-n),
                             (0.5-n), (0.5+n) );
         T1.ff *= phas; T1.fb /= phas;
         T1.bf *= phas; T1.bb /= phas;
         T = T1 * T;
     }
     
    return T.bb;
    
    return 0;
  
}


  
}}}



