#include <complex>

#include "python_globals.h"
#include <numpy/arrayobject.h>
#include <frameobject.h> // for Python traceback

#include <plask/version.h>
#include <plask/exceptions.h>
#include <plask/mesh/interpolation.h>
#include <plask/solver.h>

using namespace plask::python;

// Declare some initialization functions
namespace plask { namespace python {

void initMaterials();
void initGeometry();

void register_manager();

void register_vectors();
void register_mesh();
void register_providers();

void register_data_vectors();

void register_python_log();

void register_standard_properties();

// Hack necessary as macro import_array wants to return some value
static inline bool plask_import_array() {
    import_array1(false);
    return true;
}

// Config
Config config;
AxisNames Config::axes = AxisNames::axisNamesRegister.get("xyz");

static LoggingConfig getLoggingConfig(const Config&) {
    return LoggingConfig();
}

std::string Config::__str__() const {
    return  "axes:          " + axes_name()
          + "\nlog.color:   " + std::string(py::extract<std::string>(LoggingConfig().getLoggingColor().attr("__str__")()))
          + "\nlog.level:   " + std::string(py::extract<std::string>(py::object(maxLoglevel).attr("name")))
          + "\nlog.output:  " + std::string(py::extract<std::string>(LoggingConfig().getLoggingDest().attr("__str__")()));
    ;
}

std::string Config:: __repr__() const {
    return
        format("config.axes = '%s'", axes_name()) +
            + "\nlog.color = " + std::string(py::extract<std::string>(LoggingConfig().getLoggingColor().attr("__repr__")()))
            + "\nlog.level = LOG_" + std::string(py::extract<std::string>(py::object(maxLoglevel).attr("name")))
            + "\nlog.output = " + std::string(py::extract<std::string>(LoggingConfig().getLoggingDest().attr("__repr__")()));
    ;
}

inline static void register_config()
{
    py::class_<Config> config_class("config", "Global PLaSK configuration.", py::no_init);
    config_class
        .def("__str__", &Config::__str__)
        .def("__repr__", &Config::__repr__)
        .add_property("axes", &Config::axes_name, &Config::set_axes,
                      "String representing axis names")
        .add_property("log", &getLoggingConfig, "Settings of the logging system");
    py::scope().attr("config") = config;
}


// Globals for XML material
py::dict xml_globals;


// Print Python exception to PLaSK logging system
int printPythonException(PyObject* otype, PyObject* value, PyObject* otraceback, unsigned startline=0, const char* scriptname=nullptr, bool second_is_script=false) {
    PyTypeObject* type = (PyTypeObject*)otype;
    PyTracebackObject* original_traceback = (PyTracebackObject*)otraceback;

    if ((PyObject*)type == PyExc_SystemExit) {
        int exitcode;
        if (PyExceptionInstance_Check(value)) {
            PyObject* code = PyObject_GetAttrString(value, "code");
            if (code) { Py_DECREF(value); value = code; }
        }
        if (PyInt_Check(value))
            exitcode = (int)PyInt_AsLong(value);
        else {
            std::cerr.flush();
            std::cout.flush();
            PyObject_Print(value, stderr, Py_PRINT_RAW);
            PySys_WriteStderr("\n");
            exitcode = 1;
        }
        PyErr_Clear();
        return exitcode;
    }

    PyObject* pmessage = PyObject_Str(value);
    const char* message = py::extract<const char*>(pmessage);

    std::string error_name = type->tp_name;
    if (error_name.substr(0, 11) == "exceptions.") error_name = error_name.substr(11);

    if (original_traceback) {
        PyTracebackObject* traceback = original_traceback;
        while (traceback) {
            int lineno = traceback->tb_lineno;
            std::string filename = PyString_AsString(traceback->tb_frame->f_code->co_filename);
            if (filename == scriptname) lineno += startline;
            std::string funcname = PyString_AsString(traceback->tb_frame->f_code->co_name);
            if (funcname == "<module>" && (traceback == original_traceback || (second_is_script && traceback == original_traceback->tb_next)))
                funcname = "<script>";
            if (traceback->tb_next)
                plask::writelog(plask::LOG_ERROR_DETAIL, "%1%, line %2%, function '%3%' calling:", filename, lineno, funcname);
            else {
                if ((PyObject*)type == PyExc_IndentationError || (PyObject*)type == PyExc_SyntaxError) {
                    plask::writelog(plask::LOG_ERROR_DETAIL, "%1%, line %2%, function '%3%' calling:", filename, lineno, funcname);
                    std::string form = message;
                    std::size_t f = form.find(" (") + 2, l = form.rfind(", line ") + 7;
                    std::string msg = form.substr(0, f-2), file = form.substr(f, l-f-7);
                    try {
                        int lineno = startline + boost::lexical_cast<int>(form.substr(l, form.length()-l-1));
                        plask::writelog(plask::LOG_CRITICAL_ERROR, "%1%, line %2%: %3%: %4%", file, lineno, error_name, msg);
                    } catch (boost::bad_lexical_cast) {
                        plask::writelog(plask::LOG_CRITICAL_ERROR, "%1%: %2%", error_name, message);
                    }
                } else
                    plask::writelog(plask::LOG_CRITICAL_ERROR, "%1%, line %2%, function '%3%': %4%: %5%", filename, lineno, funcname, error_name, message);
            }
            traceback = traceback->tb_next;
        }
    } else {
        if ((PyObject*)type == PyExc_IndentationError || (PyObject*)type == PyExc_SyntaxError) {
                std::string form = message;
                std::size_t f = form.find(" (") + 2, l = form.rfind(", line ") + 7;
                std::string msg = form.substr(0, f-2), file = form.substr(f, l-f-7);
                try {
                    int lineno = startline + boost::lexical_cast<int>(form.substr(l, form.length()-l-1));
                    plask::writelog(plask::LOG_CRITICAL_ERROR, "%1%, line %2%: %3%: %4%", file, lineno, error_name, msg);
                } catch (boost::bad_lexical_cast) {
                    plask::writelog(plask::LOG_CRITICAL_ERROR, "%1%: %2%", error_name, message);
                }
        } else
            plask::writelog(plask::LOG_CRITICAL_ERROR, "%1%: %2%", error_name, message);
    }
    Py_XDECREF(pmessage);
    return 1;
}

// Default options for docstrings
py::docstring_options doc_options(
    true,   // show user defined
    true,   // show py signatures
    false   // show cpp signatures
);

}} // namespace plask::python

BOOST_PYTHON_MODULE(_plask)
{
    // Initialize numpy
    if (!plask_import_array()) throw(py::error_already_set());


    py::scope scope; // Default scope

    // Config
    register_config();

    // Log
    register_python_log();

    // Manager
    register_manager();

    // Vectors
    register_vectors();

    register_vector_of<size_t>("unsigned_int");
    register_vector_of<int>("int");
    register_vector_of<double>("float");
    register_vector_of<std::complex<double>>("complex");

    // Materials
    initMaterials();

    // Geometry
    initGeometry();

    // Meshes
    register_mesh();

    // Data vector
    register_data_vectors();

    // Solvers
    py::class_<plask::Solver, plask::shared_ptr<plask::Solver>, boost::noncopyable>("Solver", "Base class for all solvers", py::no_init)
        .add_property("name", &plask::Solver::getName, "Name of the solver object")
        .add_property("id", &plask::Solver::getId, "Id of the solver object")
        .add_property("description", &plask::Solver::getClassDescription, "Short description of the solver")
        .add_property("initialized", &plask::Solver::isInitialized, "True if the solver has been initialized")
        .def("invalidate", &plask::Solver::invalidate, "Set solver back to uninitialized state")
    ;

    // Exceptions
    register_exception<plask::Exception>(PyExc_RuntimeError);

    register_exception<plask::NotImplemented>(PyExc_NotImplementedError);
    register_exception<plask::OutOfBoundsException>(PyExc_IndexError);
    register_exception<plask::NoSuchMaterial>(PyExc_ValueError);
    register_exception<plask::NoSuchGeometryObjectType>(PyExc_TypeError);
    register_exception<plask::BadInput>(PyExc_ValueError);
    register_exception<plask::DataError>(PyExc_ValueError);
    register_exception<plask::NoValue>(PyExc_ValueError);
    register_exception<plask::NoProvider>(PyExc_TypeError);

    register_exception<plask::python::ValueError>(PyExc_ValueError);
    register_exception<plask::python::TypeError>(PyExc_TypeError);
    register_exception<plask::python::IndexError>(PyExc_IndexError);
    register_exception<plask::python::KeyError>(PyExc_KeyError);
    register_exception<plask::python::AttributeError>(PyExc_AttributeError);
    register_exception<plask::python::StopIteration>(PyExc_StopIteration);
    register_exception<plask::python::IOError>(PyExc_IOError);

    PyObject* xml_error = PyErr_NewExceptionWithDoc((char*)"plask.XMLError", (char*)"Error in XML file.", NULL, NULL);
    register_exception<plask::XMLException>(xml_error);
    py::scope().attr("XMLError") = py::handle<>(py::incref(xml_error));

    PyObject* computation_error = PyErr_NewExceptionWithDoc((char*)"plask.ComputationError", (char*)"Computational error in some PLaSK solver.",
                                                            PyExc_ArithmeticError, NULL);
    register_exception<plask::ComputationError>(computation_error);
    py::scope().attr("ComputationError") = py::handle<>(py::incref(computation_error));

    py::def("_print_exception", &printPythonException, "Print exception information to PLaSK logging system",
            (py::arg("exc_type"), "exc_value", "exc_traceback", py::arg("startline")=0, py::arg("scriptname")="", py::arg("second_is_script")=false));

    // PLaSK version
    scope.attr("version") = PLASK_VERSION;
    scope.attr("version_major") = PLASK_VERSION_MAJOR;
    scope.attr("version_minor") = PLASK_VERSION_MINOR;

    // Set global namespace for materials
    py::object numpy = py::import("numpy");
    plask::python::xml_globals = py::dict(numpy.attr("__dict__")).copy();
    plask::python::xml_globals.update(scope.attr("__dict__"));
    plask::python::xml_globals["plask"] = scope;

    scope.attr("prefix") = plask::prefixPath();
    scope.attr("lib_path") = plask::plaskLibPath();

    // Properties
    register_standard_properties();
}
