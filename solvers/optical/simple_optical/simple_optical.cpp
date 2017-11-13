#include "simple_optical.h"

namespace plask { namespace solvers { namespace simple_optical {

  

SimpleOptical::SimpleOptical(const std::string& name):plask::SolverOver<plask::Geometry2DCylindrical>(name)
{

}

void SimpleOptical::loadConfiguration(XMLReader& reader, Manager& manager) {
    // Load a configuration parameter from XML.
    // Below you have an example
    while (reader.requireTagOrEnd()) {

            parseStandardConfiguration(reader, manager, "<geometry>");
	    // axis_name = reader.getAxisLongName();
    }
}

void SimpleOptical::say_hello()
{
    std::cout<<"Hello world!!!!!!!!2222222222222 "<<std::endl;
}

  
}}}