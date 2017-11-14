#include "simple_optical.h"

namespace plask { namespace solvers { namespace simple_optical {

  

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
    std::cout << "material: "; for (double p: *points->vert()) std::cout << geometry->getMaterial(vec(0.5, p))->name() << " "; std::cout << std::endl;
    std::cout << "nr: "; for (double p: *points->vert()) std::cout << str(geometry->getMaterial(vec(0.5, p))->Nr(1300, 300)) << " "; std::cout << std::endl;
}

void SimpleOptical::simpleVerticalSolver()
{
  shared_ptr<RectangularMesh<2>> mesh = makeGeometryGrid(this->geometry->getChild());
  std::cout << "mesh: "; for (double p: *mesh->vert()) std::cout<< p << " "; std::cout << std::endl;
}

  
}}}



