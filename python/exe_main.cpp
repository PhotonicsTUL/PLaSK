#include <cmath>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
namespace py = boost::python;

#include <iostream>
#include <vector>
#include <string>

#include <plask/exceptions.h>

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
static py::object initPlaskModule(int argc, const char* argv[])
{
    // Initialize the module plask
    if (PyImport_AppendInittab("plaskcore", &PLASK_MODULE) != 0) throw plask::CriticalException("No plaskcore module");

    // Initialize Python
    Py_Initialize();

    py::object plaskcore = py::import("plaskcore");

    py::object sys = py::import("sys");
    sys.attr("modules")["plask.plaskcore"] = plaskcore;

    // Add "." to the search path
    py::list path = py::list(sys.attr("path"));
    path.insert(0, ".");
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
        initPlaskModule(argc-1, argv+1);
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
            const char* f = argv[1];
#           if PY_VERSION_HEX >= 0x03000000
                // For Python3 "from __future__ import division" flag is not necessary
                py::exec_file(f, globals);
#           else
                // We dont use py::exec_file, as we want to set "from __future__ import division" flag
                PyObject *pyfile = PyFile_FromString(const_cast<char*>(f), const_cast<char*>("r"));
                if (!pyfile) throw std::invalid_argument("No such file: " + std::string(f));
                py::handle<> file(pyfile);
                FILE *fs = PyFile_AsFile(file.get());
                PyCompilerFlags flags { CO_FUTURE_DIVISION };
                PyObject* result = PyRun_FileFlags(fs, f, Py_file_input, globals.ptr(), globals.ptr(), &flags);
                if (!result) py::throw_error_already_set();
#           endif
        } catch (std::invalid_argument err) {
            std::cerr << err.what() << "\n";
            return 100;
        } catch (py::error_already_set) {
            PyErr_Print();
            return 103;
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
