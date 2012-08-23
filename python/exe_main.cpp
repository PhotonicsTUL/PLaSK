#include <cmath>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
namespace py = boost::python;

#include <frameobject.h> // For Python traceback

#include <iostream>
#include <vector>
#include <string>
#include <stack>

#include <plask/exceptions.h>
#include <plask/utils/system.h>
#include "plask/python_manager.h"

//******************************************************************************
#ifdef __cplusplus
extern "C"
#endif

#if PY_VERSION_HEX >= 0x03000000
    PyObject* PyInit_plaskcore(void);
#   define PLASK_MODULE PyInit_plaskcore
#else
    void initplaskcore(void);
#   define PLASK_MODULE initplaskcore
#endif

// Initialize the binary modules and load the package from disc
static py::object initPlaskSolver(int argc, const char* argv[])
{
    // Initialize the plask module
    if (PyImport_AppendInittab("plaskcore", &PLASK_MODULE) != 0) throw plask::CriticalException("No plaskcore module");

    // Initialize Python
    Py_Initialize();

    py::object plaskcore = py::import("plaskcore");

    py::object sys = py::import("sys");
    sys.attr("modules")["plask.plaskcore"] = plaskcore;

    // Add search paths
    py::list path = py::list(sys.attr("path"));
    path.insert(0, "." );
    std::string plask_path = plask::prefixPath();
    plask_path += plask::FILE_PATH_SEPARATOR; plask_path += "lib";
    plask_path += plask::FILE_PATH_SEPARATOR; plask_path += "plask";
    plask_path += plask::FILE_PATH_SEPARATOR;
    path.insert(1, plask_path + "python" );
    path.insert(2, plask_path + "solvers" );
    sys.attr("path") = path;

    // Add program arguments to sys.argv
    if (argc > 0) {
        py::list sys_argv;
        for (int i = 0; i < argc; i++) {
            sys_argv.append(argv[i]);
        }
        sys.attr("argv") = sys_argv;
    }

    return plaskcore;
}

//******************************************************************************
static void from_import_all(const char* name, py::dict globals)
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
        if ((*item)[0] != '_') globals[*item] = module_dict[*item];
    }
}

//******************************************************************************
int handlePythonException() {
    if (PyErr_ExceptionMatches(PyExc_SystemExit)) return 0; // Normal exit of the program

    // Use our logging system to print exception
    PyObject* value;
    PyTypeObject* type;
    PyTracebackObject* original_traceback;

    PyErr_Fetch((PyObject**)&type, (PyObject**)&value, (PyObject**)&original_traceback);
    PyErr_NormalizeException((PyObject**)&type, (PyObject**)&value, (PyObject**)&original_traceback);

    PyObject* pmessage = PyObject_Str(value);
#   if PY_VERSION_HEX >= 0x03000000
        const char* message = py::extract<const char*>(pmessage);
#   else
        const char* message = PyString_AsString(pmessage);
#   endif

    std::string error_name = type->tp_name;
    if (error_name.substr(0, 11) == "exceptions.") error_name = error_name.substr(11);

    if (original_traceback) {
        std::stack<PyTracebackObject*> tb_stack;

        PyTracebackObject* traceback = original_traceback;
        while (traceback->tb_next != NULL) {
            tb_stack.push(traceback);
            traceback = traceback->tb_next;
        }

        int lineno = traceback->tb_lineno;
#       if PY_VERSION_HEX >= 0x03000000
            const char* filename = py::extract<const char*>(traceback->tb_frame->f_code->co_filename);
            const char* funcname = py::extract<const char*>(traceback->tb_frame->f_code->co_name);
#       else
            const char* filename = PyString_AsString(traceback->tb_frame->f_code->co_filename);
            const char* funcname = PyString_AsString(traceback->tb_frame->f_code->co_name);
#       endif
        plask::writelog(plask::LOG_CRITICAL_ERROR, "%1%, line %2%, function '%3%': %4%: %5%", filename, lineno, funcname, error_name, message);

        while (!tb_stack.empty()) {
            traceback = tb_stack.top();
            tb_stack.pop();
            int lineno = traceback->tb_lineno;
#           if PY_VERSION_HEX >= 0x03000000
                const char* filename = py::extract<const char*>(traceback->tb_frame->f_code->co_filename);
                const char* funcname = py::extract<const char*>(traceback->tb_frame->f_code->co_name);
#           else
                const char* filename = PyString_AsString(traceback->tb_frame->f_code->co_filename);
                const char* funcname = PyString_AsString(traceback->tb_frame->f_code->co_name);
#           endif
            plask::writelog(plask::LOG_DETAIL, "called from: %1%, line %2%, function '%3%'", filename, lineno, funcname);
        }

    } else {
        plask::writelog(plask::LOG_CRITICAL_ERROR, "%1%: %2%", error_name, message);
    }

    Py_XDECREF(type);
    Py_XDECREF(value);
    Py_XDECREF(pmessage);
    Py_XDECREF(original_traceback);
    return 100;
}

//******************************************************************************
int main(int argc, const char *argv[])
{
    // Test if we want to import plask into global namespace
    bool from_import = true;
    bool force_interactive = false;
    while (argc > 1) {
        if (std::string(argv[1]) == "-n") {
            from_import = false;
            --argc; ++argv;
        } else if (std::string(argv[1]) == "-i") {
            force_interactive = true;
            --argc; ++argv;
        } else break;
    }

    // Initalize python and load the plask module
    try {
        initPlaskSolver(argc-1, argv+1);
    } catch (plask::CriticalException) {
        std::cerr << "CriticalError: Cannot import plask builtin module.\n";
        return 101;
    } catch (py::error_already_set) {
        PyErr_Print();
        return 102;
    }

    // Test if we should use the file or start an interactive mode
    if(argc > 1 && !force_interactive) { // load commands from file

        // Add plask to the global namespace
        py::object plask = py::import("plask");
        py::dict globals = py::dict(py::import("__main__").attr("__dict__"));

        plask.attr("_globals_") = globals;

        globals["plask"] = plask; // import plask
        if (from_import) { // from plask import *
            from_import_all("plask", globals);
        }

        try {
            std::string filename = argv[1];
            boost::optional<bool> xml_input;

            // Detect if the file is Python script or PLaSK input

            // check file extension
            std::string ext = filename.substr(filename.length()-4);
            if (ext == ".xpl") xml_input.reset(true);
            else if (ext == ".xml") xml_input.reset(true);
            else if (ext.substr(2) == ".py") xml_input.reset(false);

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
                manager->loadFromFILE(file);
                std::fclose(file);
                // manager->script = plask::python::PythonManager::removeSpaces(manager->script);
                plask::python::PythonManager::export_dict(globals["__manager__"], globals);

                PyObject* result = NULL;
#               if PY_VERSION_HEX >= 0x03000000
                    PyObject* code = Py_CompileString(manager->script.c_str(), (filename+", tag <script>").c_str(), Py_file_input);
                    if (code)
                        result = PyEval_EvalCode(code, globals.ptr(), globals.ptr());
#               else
                    PyCompilerFlags flags { CO_FUTURE_DIVISION };
                    PyObject* code = Py_CompileStringFlags(manager->script.c_str(), (filename+", tag <script>").c_str(), Py_file_input, &flags);
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
        } catch (std::invalid_argument err) {
            plask::writelog(plask::LOG_CRITICAL_ERROR, err.what());
            return 1;
        } catch (plask::Exception err) {
            plask::writelog(plask::LOG_CRITICAL_ERROR, err.what());
            return 2;
        } catch (plask::XMLException err) {
            plask::writelog(plask::LOG_CRITICAL_ERROR, "'%1%': %2%", argv[1], err.what());
            return 3;
        } catch (py::error_already_set) {
            return handlePythonException();
        }

    } else { // start the interactive console

        try {
            py::object interactive = py::import("plask.interactive");
            interactive.attr("_import_all_") = from_import;
            py::list sys_argv;
            if (argc == 1) sys_argv.append("");
            for (int i = 1; i < argc; i++) sys_argv.append(argv[i]);
            interactive.attr("interact")(py::object(), sys_argv);
        } catch (py::error_already_set) {
            PyErr_Print();
            return 104;
        }
    }

    // Close the Python interpreter and exit
    return 0;
}
