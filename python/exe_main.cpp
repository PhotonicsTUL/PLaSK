
#include <Python.h>

#include <iostream>
using namespace std;

const char* BANNER = "You are entering the interactive mode of PLaSK.\n"
                     "Module plask is already imported into global namespace.";

//******************************************************************************
#ifdef __cplusplus
extern "C"
#endif
void initplask_(void);

// Initialize the binary modules and load the package from disc
PyObject* initPython(int argc, char* argv[])
{
    // Initialize the module plask
    if (PyImport_AppendInittab("plask_", &initplask_) != 0) return NULL;

    // Initialize Python
    Py_Initialize();

    PyObject* pSys = PyImport_ImportModule("sys");

    // Add "." to the search path
    PyObject* pPath = PyDict_GetItemString(PyModule_GetDict(pSys), "path");
    PyObject* pDir = PyString_FromString("");
    PyList_Insert(pPath, 0, pDir);
    Py_DECREF(pDir);

    // Add program arguments to sys.argv
    if (argc > 0) {
        PyObject* pArgs = PyList_New(argc);
        for (int i = 0; i < argc; i++) {
            PyObject* arg = PyString_FromString(argv[i]);
            PyList_SET_ITEM(pArgs, i, arg);
        }
        PyDict_SetItemString(PyModule_GetDict(pSys), "argv", pArgs);
        Py_DECREF(pArgs);
    }

    Py_DECREF(pSys);

    // Hack to make sure the python part load the built-in module and not the lib file
    PyObject* pLibplask = PyImport_ImportModule("plask_");
    PyObject* pModules = PyImport_GetModuleDict();
    PyDict_SetItemString(pModules, "plask_", pLibplask);
    PyDict_DelItemString(pModules, "plask_");
    Py_DECREF(pLibplask);

    // Load the Python part
    PyObject* pName = PyString_FromString("plask");
    PyObject* pModule = PyImport_Import(pName);
    // TODO: from plask import *
    Py_DECREF(pName);

    return pModule;
}

#include <vector>

//******************************************************************************
int main(int argc, char *argv[])
{
    // Initalize python and load the mol3d module
    PyObject* pModule = initPython(argc-1, argv+1);
    if (pModule == NULL) {
        PyErr_Print();
        Py_Finalize();
        return 1;
    }

    // Test if we should use the file or start an interactive mode
    if(argc > 1) { // We load the commands from file
        FILE* file;

        file = fopen(argv[1], "r");
        if (file) {
            PyRun_SimpleFile(file, argv[1]);
            fclose(file);
        } else {
            std::cerr << "FileError: Could not open file " << argv[1] << "\n";
            Py_DECREF(pModule);
            Py_Finalize();
            return 1;
        }
    } else { // Start the interactive mode
        PyObject* pCode = PyImport_ImportModule("code");
        if (pCode == NULL) {
            PyErr_Clear();
            std::cerr << "Error: Cannot init Python console" << argv[1] << "\n";
            Py_Finalize();
            return 1;
        }

        PyObject* locals = PyDict_New();
        PyDict_SetItem(locals, PyString_FromString("plask"), pModule);

        PyObject* pInteract = PyDict_GetItemString(PyModule_GetDict(pCode), "interact");
        PyObject_CallFunction(pInteract, (char*)"sOO", BANNER, Py_None, locals);

        Py_DECREF(pCode);
    }

    // Close the Python interpreter and exit
    Py_DECREF(pModule);
    Py_Finalize();
    return 0;
}
