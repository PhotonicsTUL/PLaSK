#include <cmath>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
namespace py = boost::python;

#include <cstdio>
#include <vector>
#include <string>
#include <stack>

#include <plask/version.h>
#include <plask/exceptions.h>
#include <plask/utils/system.h>
#include <plask/log/log.h>
#include <plask/python_globals.h>
#include <plask/python_manager.h>
#include <plask/utils/string.h>
#include <plask/license/verify.h>

//******************************************************************************
#if PY_VERSION_HEX >= 0x03000000
    extern "C" PyObject* PyInit__plask(void);
#   define PLASK_MODULE PyInit__plask
    inline auto PyString_Check(PyObject* o) -> decltype(PyUnicode_Check(o)) { return PyUnicode_Check(o); }
    inline const char* PyString_AsString(PyObject* o) { return py::extract<const char*>(o); }
    inline bool PyInt_Check(PyObject* o) { return PyLong_Check(o); }
    inline long PyInt_AsLong(PyObject* o) { return PyLong_AsLong(o); }
#else
    extern "C" void init_plask(void);
#   define PLASK_MODULE init_plask
#endif

//******************************************************************************

// static PyThreadState* mainTS;   // state of the main thread

namespace plask { namespace python {

    int printPythonException(PyObject* otype, py::object value, PyObject* otraceback, const char* scriptname=nullptr, bool second_is_script=false);

    void createPythonLogger();
}}

void showError(const std::string& msg, const std::string& cap="Error");

//******************************************************************************
// Initialize the binary modules and load the package from disk
static py::object initPlask(int argc, const char* argv[])
{
    // Initialize the plask module
    if (PyImport_AppendInittab("_plask", &PLASK_MODULE) != 0) throw plask::CriticalException("No _plask module");

    // Initialize Python
    Py_Initialize();

    // Add search paths
    py::object sys = py::import("sys");
    py::list path = py::list(sys.attr("path"));
    std::string plask_path = plask::prefixPath();
    plask_path += plask::FILE_PATH_SEPARATOR; plask_path += "lib";
    plask_path += plask::FILE_PATH_SEPARATOR; plask_path += "plask";
    path.insert(0, plask_path);
    std::string solvers_path = plask_path;
    plask_path += plask::FILE_PATH_SEPARATOR; plask_path += "python";
    solvers_path += plask::FILE_PATH_SEPARATOR; solvers_path += "solvers";
    path.insert(0, plask_path);
    path.insert(1, solvers_path);
    if (argc > 0) // This is correct!!! argv[0] here is argv[1] in `main`
        try {
            path.insert(0, boost::filesystem::absolute(boost::filesystem::path(argv[0])).parent_path().string());
        } catch (std::runtime_error) { // can be thrown if there is wrong locale set
            std::string file(argv[0]);
            size_t pos = file.rfind(plask::FILE_PATH_SEPARATOR);
            if (pos == std::string::npos) pos = 0;
            path.insert(0, file.substr(0, pos));
        }
    else
        path.insert(0, "");

    sys.attr("path") = path;

    sys.attr("executable") = plask::exePathAndName();

    py::object _plask = py::import("_plask");

    sys.attr("modules")["plask._plask"] = _plask;

    // Add program arguments to sys.argv
    if (argc > 0) {
        py::list sys_argv;
        for (int i = 0; i < argc; i++) {
            sys_argv.append(argv[i]);
        }
        sys.attr("argv") = sys_argv;
    }

    // mainTS = PyEval_SaveThread();
    //PyEval_ReleaseLock();

    return _plask;
}

//******************************************************************************
int handlePythonException(const char* scriptname=nullptr) {
    PyObject* value;
    PyObject* type;
    PyObject* original_traceback;
    PyErr_Fetch(&type, &value, &original_traceback);
    PyErr_NormalizeException(&type, &value, &original_traceback);
    py::handle<> value_h(value), type_h(type), original_traceback_h(py::allow_null(original_traceback));
    if (type == PyExc_SystemExit) {
        int exitcode = 0;
        if (PyInt_Check(value)) exitcode = (int)PyInt_AsLong(value);
        PyErr_Clear();
        return exitcode;
    }
    std::string msg = py::extract<std::string>(py::str(value_h));
    std::string cap = py::extract<std::string>(py::object(type_h).attr("__name__"));
    showError(msg, cap);
    return 1;
}


//******************************************************************************
// Finalize Python interpreter
void endPlask() {
    // PyEval_RestoreThread(mainTS);
    Py_Finalize(); // Py_Finalize is not supported by Boost
}


//******************************************************************************
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#include <plask/utils/minimal_winows.h>
#include <boost/tokenizer.hpp>
//#define BOOST_USE_WINDOWS_H

void showError(const std::string& msg, const std::string& cap) {
    MessageBox(NULL, msg.c_str(), ("PLaSK - " + cap).c_str(), MB_OK | MB_ICONERROR);
}

int WinMain(HINSTANCE, HINSTANCE, LPSTR cmdline, int)
{
    std::string command_line(cmdline);
    boost::tokenizer<boost::escaped_list_separator<char>> tokenizer(command_line, boost::escaped_list_separator<char>('^', ' ', '"'));
    std::deque<std::string> args(tokenizer.begin(), tokenizer.end());
    args.push_front(plask::exePathAndName());
    int argc = args.size();
    const char* argv[argc];
    const char** dst = argv; for(const auto& src: args) { *(dst++) = src.c_str(); }
#else

void showError(const std::string& msg, const std::string& cap) {
    plask::writelog(plask::LOG_CRITICAL_ERROR, cap + ": " + msg);
}

int main(int argc, const char *argv[])
{
#endif
    // Set the Python logger
    // plask::python::createPythonLogger();
    plask::createDefaultLogger();

    // Initalize python and load the plask module
    try {
        initPlask(argc, argv);
    } catch (plask::CriticalException) {
        showError("Cannot import plask builtin module.");
        endPlask();
        return 101;
    } catch (py::error_already_set) {
        handlePythonException();
        endPlask();
        return 102;
    }

    // Import and run GUI
    try {
        py::object gui = py::import("gui");
        gui.attr("main")();
    } catch (std::invalid_argument& err) {
        showError(err.what(), "Invalid argument");
        endPlask();
        return -1;
    } catch (plask::Exception& err) {
        showError(err.what());
        endPlask();
        return 3;
    } catch (py::error_already_set) {
        int exitcode = handlePythonException(argv[0]);
        endPlask();
        return exitcode;
    } catch (std::runtime_error& err) {
        showError(err.what());
        endPlask();
        return 3;
    }

    // Close the Python interpreter and exit
    endPlask();
    return 0;
}
