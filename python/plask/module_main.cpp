#include <boost/python.hpp>
namespace python = boost::python;

const char* test () {
    return "test";
}

BOOST_PYTHON_MODULE(plask_)
{
    python::def("test", &test);
}
