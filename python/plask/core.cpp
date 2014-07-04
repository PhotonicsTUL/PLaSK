#include <complex>

#include <boost/algorithm/string.hpp>

#include "python_globals.h"
#include "python_numpy.h"
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

std::string getPythonExceptionMessage() {
    PyObject *value, *type, *original_traceback;
    PyErr_Fetch(&type, &value, &original_traceback);
    PyErr_NormalizeException(&type, &value, &original_traceback);
    py::handle<> value_h(value), type_h(type), original_traceback_h(py::allow_null(original_traceback));
    return py::extract<std::string>(py::str(value_h));
}

// Config
PLASK_PYTHON_API AxisNames current_axes = AxisNames::axisNamesRegister.get("zxy");

static LoggingConfig getLoggingConfig(const Config&) {
    return LoggingConfig();
}

std::string Config::axes_name() const {
    return current_axes.str();
}
void Config::set_axes(std::string axis) {
    current_axes = AxisNames::axisNamesRegister.get(axis);
}

std::string Config::__str__() const {
    return  "axes:        " + axes_name()
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
    py::class_<Config> config_class("config",

        "Global PLaSK configuration.\n\n"

        "This class has only one instance and it contains global configuration options.\n"
        "The attributes of this class are config parameters that can be set using the\n"
        "``config`` object.\n\n"

        "Example:\n"
        "    >>> config.axes = 'xy'\n"
        "    >>> config.log.level = 'debug'\n"
        "    >>> print config\n"
        "    axes:        zxy\n"
        "    log.color:   ansi\n"
        "    log.level:   DEBUG\n"
        "    log.output:  stdout\n"

        , py::no_init);
    config_class
        .def("__str__", &Config::__str__)
        .def("__repr__", &Config::__repr__)
        .add_property("axes", &Config::axes_name, &Config::set_axes,
                      "String representing axis names.\n\n"

                      "The accepted values are listed below. Each row shows different aliases for\n"
                      "the same axes:\n\n"

                      "================ ================ ================\n"
                      "`xyz`            `yz`             `z_up`\n"
                      "`zxy`            `xy`             `y_up`\n"
                      "`prz`            `rz`             `rad`\n"
                      "`ltv`                             `abs`\n"
                      "`long,tran,vert`                  `absolute`\n"
                      "================ ================ ================\n\n"

                      "The choice of the axes should depend on your structure. In Cartesian coordinates\n"
                      "you probablye prefer `xyz` or `zxy`. In cylindrical ones the most natural choice\n"
                      "is `prz`. However, it is important to realize that any names can be chosen in\n"
                      "any geometry and they are fully independent from it.\n"
                     )
        .add_property("log", &getLoggingConfig,
                      "Settings of the logging system.\n\n"

                      "This setting has several subattributes listed below:\n\n"

                      "**color**\n"
                      "        System used for coloring the log messages depending on their level.\n"
                      "        This parameter can have on of the following values:\n\n"

                      "        :ansi:    Use ANSI codes for coloring. Works best in UNIX-like systems\n"
                      "                  (Linux, OSX) or with GUI launchers.\n"
                      "        :windows: Use Windows API for coloring. Availale only on Windows.\n"
                      "        :none:    Do not perform coloring at all. Recomended when redirecting\n"
                      "                  output to a file.\n\n"

                      "        On its start PLaSK tries to automatically determin the best value for\n"
                      "        this option, so usually you will not need to change it.\n\n"

                      "**level**\n"
                      "        Maximum logging level. It can be one of:\n\n"

                      "        :CRITICAL_ERROR: Critical errors that result in program interruption.\n"
                      "        :ERROR:          Minor errors that do not break the whole program flow.\n"
                      "        :ERROR_DETAIL:   Details of the errors with more information on them.\n"
                      "        :WARNING:        Important warnings that you investiagate.\n"
                      "        :INFO:           General information of the executed operations.\n"
                      "        :RESULT:         Some intermediate computations results.\n"
                      "        :DATA:           Some data used for tracking the computations.\n"
                      "        :DETAIL:         Details of computations processes.\n"
                      "        :DEBUG:          Additional information useful for debugging PLaSK.\n\n"

                      "        Setting any of the above levels will instruct PLaSK to print only\n"
                      "        information of the specified level and above. It is recommended to\n"
                      "        always set the logging level at least to 'WARNING'.\n\n"

                      "**output**\n"
                      "        Stream to which the log messages are prited. Can be either **stderr**\n"
                      "        (which is the default) or **stdout** (turned on for interactive mode).\n\n"

                      "Usually you should only want to change the :attr:`config.log.level` value.\n"
                      "However this setting is ignored when the plask option :option:`-l`\n"
                      "is specified.\n"
                     );
    py::scope().attr("config") = Config();
}


// Globals for XML material
py::dict xml_globals;


// Print Python exception to PLaSK logging system
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
__declspec(dllexport)
#endif
int printPythonException(PyObject* otype, py::object value, PyObject* otraceback, unsigned startline=0, const char* scriptname=nullptr, bool second_is_script=false) {
    PyTypeObject* type = (PyTypeObject*)otype;
    PyTracebackObject* original_traceback = (PyTracebackObject*)otraceback;

    if ((PyObject*)type == PyExc_SystemExit) {
        int exitcode;
        if (PyExceptionInstance_Check(value.ptr())) {
            PyObject* code = PyObject_GetAttrString(value.ptr(), "code");
            if (code) { value = py::object(py::handle<>(code)); }
        }
        if (PyInt_Check(value.ptr()))
            exitcode = (int)PyInt_AsLong(value.ptr());
        else {
            std::cerr.flush();
            std::cout.flush();
            PyObject_Print(value.ptr(), stderr, Py_PRINT_RAW);
            PySys_WriteStderr("\n");
            exitcode = 1;
        }
        PyErr_Clear();
        return exitcode;
    }


    std::string message = py::extract<std::string>(py::str(value));
    boost::replace_all(message, "\n", "\n                ");

    std::string error_name = type->tp_name;
    if (error_name.substr(0, 11) == "exceptions.") error_name = error_name.substr(11);

    if (original_traceback) {
        PyTracebackObject* traceback = original_traceback;
        while (traceback) {
            int lineno = traceback->tb_lineno;
            std::string filename = PyString_AsString(traceback->tb_frame->f_code->co_filename);
            int flineno = (scriptname != nullptr && filename == scriptname)? startline + lineno : lineno;
            std::string scriptline = (lineno != flineno)? format(" (%1% in script)", lineno) : "";
            std::string funcname = PyString_AsString(traceback->tb_frame->f_code->co_name);
            if (funcname == "<module>" && (traceback == original_traceback || (second_is_script && traceback == original_traceback->tb_next)))
                funcname = "<script>";
            if (traceback->tb_next)
                plask::writelog(plask::LOG_ERROR_DETAIL, "%1%, line %2%%3%, function '%4%' calling:", filename, flineno, scriptline, funcname);
            else {
                if ((PyObject*)type == PyExc_IndentationError || (PyObject*)type == PyExc_SyntaxError) {
                    plask::writelog(plask::LOG_ERROR_DETAIL, "%1%, line %2%%3%, function '%4%' calling:", filename, flineno, scriptline, funcname);
                    std::string form = message;
                    std::size_t f = form.find(" (") + 2, l = form.rfind(", line ") + 7;
                    std::string msg = form.substr(0, f-2), file = form.substr(f, l-f-7);
                    try {
                        int lineno = boost::lexical_cast<int>(form.substr(l, form.length()-l-1));
                        int flineno = startline + lineno;
                        std::string scriptline = (lineno != flineno)? format(" (%1% in script)", lineno) : "";
                        plask::writelog(plask::LOG_CRITICAL_ERROR, "%1%, line %2%%3%: %4%: %5%", file, flineno, scriptline, error_name, msg);
                    } catch (boost::bad_lexical_cast) {
                        plask::writelog(plask::LOG_CRITICAL_ERROR, "%1%: %2%", error_name, message);
                    }
                } else
                    plask::writelog(plask::LOG_CRITICAL_ERROR, "%1%, line %2%%3%, function '%4%': %5%: %6%", filename, flineno, scriptline, funcname, error_name, message);
            }
            traceback = traceback->tb_next;
        }
    } else {
        if ((PyObject*)type == PyExc_IndentationError || (PyObject*)type == PyExc_SyntaxError) {
                std::string form = message;
                std::size_t f = form.find(" (") + 2, l = form.rfind(", line ") + 7;
                std::string msg = form.substr(0, f-2), file = form.substr(f, l-f-7);
                try {
                    int lineno = boost::lexical_cast<int>(form.substr(l, form.length()-l-1));
                    int flineno = startline + lineno;
                    std::string scriptline = (lineno != flineno)? format(" (%1% in script)", lineno) : "";
                    plask::writelog(plask::LOG_CRITICAL_ERROR, "%1%, line %2%%3%: %4%: %5%", file, flineno, scriptline, error_name, msg);
                } catch (boost::bad_lexical_cast) {
                    plask::writelog(plask::LOG_CRITICAL_ERROR, "%1%: %2%", error_name, message);
                }
        } else
            plask::writelog(plask::LOG_CRITICAL_ERROR, "%1%: %2%", error_name, message);
    }
    return 1;
}

// Default options for docstrings
py::docstring_options doc_options(
    true,   // show user defined
    true,   // show py signatures
    false   // show cpp signatures
);

}} // namespace plask::python

#ifdef PRINT_STACKTRACE_ON_EXCEPTION
#   if defined(_WIN32) || defined(__WIN32__) || defined(WIN32) //win32 support
#       include <win_printstack.hpp>
#   else
#       include <backward.hpp>
        void printStack(void) {
            backward::StackTrace st;
            st.load_here(256);
            backward::Printer printer;
            printer.address = true;
            printer.print(st, stderr);
        }
#   endif
#endif

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
    py::class_<plask::Solver, plask::shared_ptr<plask::Solver>, boost::noncopyable>
    solver("Solver", "Base class for all solvers.", py::no_init);
    solver
        .add_property("id", &plask::Solver::getId,
                      "Id of the solver object. (read only)\n\n"
                      "Example:\n"
                      "    >>> mysolver.id\n"
                      "    mysolver:category.type")
        .add_property("initialized", &plask::Solver::isInitialized,
                      "True if the solver has been initialized. (read only)\n\n"
                      "Solvers usually get initialized at the beginning of the computations.\n"
                      "You can clean the initialization state and free the memory by calling\n"
                      "the :meth:`invalidate` method.")
        .def("invalidate", &plask::Solver::invalidate,
             "Set the solver back to uninitialized state.\n\n"
             "This method frees the memory allocated by the solver and sets\n"
             ":attr:`initialized` to *False*.")
    ;
    solver.attr("__module__") = "plask";

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

#   ifdef PRINT_STACKTRACE_ON_EXCEPTION
        py::def("_print_stack", &printStack, "Print C stack (for debug purposes_");
#   endif

    // PLaSK version
    scope.attr("version") = PLASK_VERSION;
    scope.attr("version_major") = PLASK_VERSION_MAJOR;
    scope.attr("version_minor") = PLASK_VERSION_MINOR;

    // Set global namespace for materials
    py::object numpy = py::import("numpy");
    plask::python::xml_globals = py::dict(numpy.attr("__dict__")).copy();
    plask::python::xml_globals.update(scope.attr("__dict__"));
    plask::python::xml_globals["plask"] = scope;
    py::incref(plask::python::xml_globals.ptr()); // HACK: Prevents segfault on exit. I don't know why it is needed.

    scope.attr("prefix") = plask::prefixPath();
    scope.attr("lib_path") = plask::plaskLibPath();

    // Properties
    register_standard_properties();

    plask::writelog(plask::LOG_INFO, PLASK_BANNER);
    plask::writelog(plask::LOG_INFO, PLASK_COPYRIGHT);
}
