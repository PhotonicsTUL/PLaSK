#include <cmath>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
namespace py = boost::python;

#include <iostream>
#include <vector>
#include <string>
#include <stack>

#include <plask/exceptions.h>
#include <plask/utils/system.h>
#include <plask/log/log.h>
#include "plask/python_globals.h"
#include "plask/python_manager.h"

#ifdef _WIN32
#   include <windows.h>
#endif

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

py::dict globals;


//******************************************************************************
// static PyThreadState* mainTS;   // state of the main thread
namespace plask { namespace python {

    int printPythonException(PyObject* otype, PyObject* value, PyObject* otraceback, unsigned startline=0, bool second_is_script=false);

    shared_ptr<Logger> makePythonLogger();

}}

//******************************************************************************
static void from_import_all(const char* name, py::dict& dest)
{
    py::object module = py::import(name);
    py::dict module_dict = py::dict(module.attr("__dict__"));
    py::list all;

    try {
        all = py::list(module.attr("__all__"));
    } catch (py::error_already_set) {
        PyErr_Clear();
        all = module_dict.keys();
    }
    py::stl_input_iterator<std::string> begin(all), end;
    for (auto item = begin; item != end; item++) {
        if ((*item)[0] != '_') dest[*item] = module_dict[*item];
    }
}

//******************************************************************************
// Initialize the binary modules and load the package from disc
static py::object initPlask(int argc, const char* argv[])
{
    // Initialize the plask module
    if (PyImport_AppendInittab("_plask", &PLASK_MODULE) != 0) throw plask::CriticalException("No _plask module");

    // Initialize Python
    Py_Initialize();

    // Add search paths
    py::object sys = py::import("sys");
    py::list path = py::list(sys.attr("path"));
    path.insert(0, "." );
    std::string plask_path = plask::prefixPath();
    plask_path += plask::FILE_PATH_SEPARATOR; plask_path += "lib";
    plask_path += plask::FILE_PATH_SEPARATOR; plask_path += "plask";
    plask_path += plask::FILE_PATH_SEPARATOR; plask_path += "python";
    path.insert(1, plask_path);
    sys.attr("path") = path;

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
// This functions closes all matplotlib windows in order to avoid the annoying
// 'Fatal Python error: PyEval_RestoreThread: NULL tstate' bug on Windows.
static inline void fixMatplotlibBug() {
#if defined(_WIN32)
    py::object modules = py::import("sys").attr("modules");
    if (py::dict(modules).has_key("matplotlib")) {
        try {
            py::object pylab = py::import("pylab");
            std::string backend = py::extract<std::string>(pylab.attr("get_backend")());
            if (backend == "TkAgg") pylab.attr("close")("all"); // fix bug in TkAgg backend in Windows
        } catch (py::error_already_set) {
            PyErr_Clear(); // silently ignore errors here
        }
    }
#endif
}


//******************************************************************************
int handlePythonException(unsigned startline=0) {
    PyObject* value;
    PyObject* type;
    PyObject* original_traceback;

    PyErr_Fetch(&type, &value,&original_traceback);
    PyErr_NormalizeException(&type, &value, &original_traceback);

    int retval = plask::python::printPythonException(type, value, original_traceback, startline);

    Py_XDECREF(type);
    Py_XDECREF(value);
    Py_XDECREF(original_traceback);

    return retval;
}


//******************************************************************************
// Finalize Python interpreter
void endPlask() {
    // PyEval_RestoreThread(mainTS);
    fixMatplotlibBug();
    // Py_Finalize(); // Py_Finalize is not supported by Boost
}


//******************************************************************************
int main(int argc, const char *argv[])
{
#   ifdef _WIN32
        SetDllDirectory(plask::exePath().c_str());
#   endif

    // Test if we want to import plask into global namespace
    bool from_import = true;
    bool force_interactive = false;
    const char* command = nullptr;

    while (argc > 1) {
        if (std::string(argv[1]) == "-n") {
            from_import = false;
            --argc; ++argv;
        } else if (std::string(argv[1]) == "-i") {
            force_interactive = true;
            --argc; ++argv;
        } else if (std::string(argv[1]) == "-c") {
            command = argv[2];
            argv[2] = "-c";
            --argc; ++argv;
            break;
        } else break;
    }

    // Initalize python and load the plask module
    try {
        initPlask(argc-1, argv+1);
    } catch (plask::CriticalException) {
        plask::writelog(plask::LOG_CRITICAL_ERROR, "Cannot import plask builtin module.");
        endPlask();
        return 101;
    } catch (py::error_already_set) {
        handlePythonException();
        endPlask();
        return 102;
    }

    // Set the Python logger
    plask::default_logger = plask::python::makePythonLogger();

    // Test if we should run commans specified in the command line, use the file or start an interactive mode
    if (command) { // run command specified in the command line

        try {
            globals = py::dict(py::import("__main__").attr("__dict__"));
            py::object plask = py::import("plask");
            plask.attr("__globals") = globals;
            globals["plask"] = plask;                           // import plask
            if (from_import) from_import_all("plask", globals); // from plask import *

            PyObject* result = NULL;
#           if PY_VERSION_HEX >= 0x03000000
                PyObject* code = Py_CompileString(command, "-c", Py_file_input);
                if (code) result = PyEval_EvalCode(code, globals.ptr(), globals.ptr());
#           else
                PyCompilerFlags flags { CO_FUTURE_DIVISION };
                PyObject* code = Py_CompileStringFlags(command, "-c", Py_file_input, &flags);
                if (code) result = PyEval_EvalCode((PyCodeObject*)code, globals.ptr(), globals.ptr());
#           endif
            Py_XDECREF(code);
            if (!result) py::throw_error_already_set();
            else Py_DECREF(result);

        } catch (py::error_already_set) { // This should not happen
            int exitcode = handlePythonException();
            endPlask();
            return exitcode;
        } catch (...) {
            return 0;
        }

    } else if(argc > 1 && !force_interactive && argv[1][0] != 0) { // load commands from file

        globals = py::dict(py::import("__main__").attr("__dict__"));
        // py::incref(globals.ptr());

        // Add plask to the global namespace
        py::object plask = py::import("plask");
        plask.attr("__globals") = globals;
        globals["plask"] = plask;                           // import plask
        if (from_import) from_import_all("plask", globals); // from plask import *

        unsigned scriptline = 0;

        try {
            std::string filename = argv[1];
            boost::optional<bool> xml_input;

            globals["__file__"] = filename;

            // Detect if the file is Python script or PLaSK input

            // check file extension
            try {
                std::string ext = filename.substr(filename.length()-4);
                if (ext == ".xpl") xml_input.reset(true);
                else if (ext == ".xml") xml_input.reset(true);
                else if (ext.substr(2) == ".py") xml_input.reset(false);
            } catch (std::out_of_range) {}

            if (!xml_input) {
                // check first char (should be '<' in XML)
                FILE* file = std::fopen(filename.c_str(), "r");
                if (!file) throw std::invalid_argument("No such file: '" + filename + "'");
                int c;
                while ((c = std::getc(file))) {
                    if (!std::isspace(c) || c == EOF) break;
                }
                std::fclose(file);
                if (c == '<') xml_input.reset(true);
                else xml_input.reset(false);
            }

            if (*xml_input) {
                auto manager = plask::make_shared<plask::python::PythonManager>();
                globals["__manager__"] = py::object(manager);
                FILE* file = std::fopen(filename.c_str(), "r");
                if (!file) throw std::invalid_argument("No such file: " + filename);
                manager->loadFromFILE(file); // it closes the file
                scriptline = manager->scriptline;
                // manager->script = plask::python::PythonManager::removeSpaces(manager->script);
                plask::python::PythonManager::export_dict(globals["__manager__"], globals);

                // Set default axes if all loaded geometries share the same
                boost::optional<plask::AxisNames> axes;
                for (const auto& geometry: manager->roots) {
                    if (!axes) axes.reset(geometry->axisNames);
                    else if (geometry->axisNames != *axes) {
                        axes.reset();
                        break;
                    }
                }
                if (axes) plask::python::config.axes = *axes;

                PyObject* result = NULL;
#               if PY_VERSION_HEX >= 0x03000000
                    PyObject* code = Py_CompileString(manager->script.c_str(), filename.c_str(), Py_file_input);
                    if (code)
                        result = PyEval_EvalCode(code, globals.ptr(), globals.ptr());
#               else
                    PyCompilerFlags flags { CO_FUTURE_DIVISION };
                    PyObject* code = Py_CompileStringFlags(manager->script.c_str(), filename.c_str(), Py_file_input, &flags);
                    if (code)
                        result = PyEval_EvalCode((PyCodeObject*)code, globals.ptr(), globals.ptr());
#               endif
                Py_XDECREF(code);
                if (!result) py::throw_error_already_set();
                else Py_DECREF(result);

            } else {
#               if PY_VERSION_HEX >= 0x03000000
                    PyObject* pyfile = PyUnicode_FromString(filename.c_str());
                    FILE* file = _Py_fopen(pyfile, "r");
                    PyObject* result = PyRun_File(file, filename.c_str(), Py_file_input, globals.ptr(), globals.ptr());
                    fclose(file);
#               else
                    // We want to set "from __future__ import division" flag
                    PyObject *pyfile = PyFile_FromString(const_cast<char*>(filename.c_str()), const_cast<char*>("r"));
                    if (!pyfile) throw std::invalid_argument("No such file: '" + filename + "'");
                    FILE* file = PyFile_AsFile(pyfile);
                    PyCompilerFlags flags { CO_FUTURE_DIVISION };
                    PyObject* result = PyRun_FileFlags(file, filename.c_str(), Py_file_input, globals.ptr(), globals.ptr(), &flags);
#               endif
                Py_DECREF(pyfile);
                if (!result) py::throw_error_already_set();
                else Py_DECREF(result);
            }

        } catch (std::invalid_argument& err) {
            plask::writelog(plask::LOG_CRITICAL_ERROR, err.what());
            endPlask();
            return -1;
        } catch (plask::XMLException& err) {
            plask::writelog(plask::LOG_CRITICAL_ERROR, "%1%: XMLError: %2%", argv[1], err.what());
            endPlask();
            return 2;
        } catch (plask::Exception& err) {
            plask::writelog(plask::LOG_CRITICAL_ERROR, err.what());
            endPlask();
            return 3;
        } catch (py::error_already_set) {
            int exitcode = handlePythonException(scriptline);
            endPlask();
            return exitcode;
        }

    } else { // start the interactive console

        py::object sys = py::import("sys");
        sys.attr("executable") = plask::exePathAndName();

        try {
            py::object interactive = py::import("plask.interactive");
            interactive.attr("_import_all_") = from_import;
            py::list sys_argv;
            if (argc == 1) sys_argv.append("");
            for (int i = 1; i < argc; i++) sys_argv.append(argv[i]);
            interactive.attr("interact")(py::object(), sys_argv);
        } catch (py::error_already_set) { // This should not happen
            int exitcode = handlePythonException();
            endPlask();
            return exitcode;
        } catch (...) {
            return 0;
        }
    }

    // Close the Python interpreter and exit
    endPlask();
    return 0;
}
