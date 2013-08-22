#include <windows.h>
#include <Python.h>

#define IDR_MYTEXTFILE 101
#define TEXTFILE 256

void LoadFileInResource(int name, int type, DWORD& size, const char*& data)
{
    HMODULE handle = ::GetModuleHandle(NULL);
    HRSRC rc = ::FindResource(handle, MAKEINTRESOURCE(name), MAKEINTRESOURCE(type));
    HGLOBAL rcData = ::LoadResource(handle, rc);
    size = ::SizeofResource(handle, rc);
    data = static_cast<const char*>(::LockResource(rcData));
}

// Usage example
int main(int argc, char** argv)
{
    DWORD size = 0;
    const char* data = NULL;
    LoadFileInResource(IDR_MYTEXTFILE, TEXTFILE, size, data);
    // The text stored in the resource might not be NULL terminated.
    char* buffer = new char[size+1];
    ::memcpy(buffer, data, size);
    buffer[size] = 0; // NULL terminator
    
    Py_Initialize();
    PySys_SetArgvEx(argc, argv, 0);
    PyObject* code = Py_CompileString(buffer, argv[0], Py_file_input);
    delete[] buffer;

    int ok = 0;

    if (code) {
        PyObject* globals = PyModule_GetDict(PyImport_AddModule("__main__"));
        PyObject* result = PyEval_EvalCode((PyCodeObject*)code, globals, globals);
        if (!result) {
           PyErr_Print();
           ok = 1;
        }
        Py_XDECREF(result);
        Py_XDECREF(code);
    } else {
        PyErr_Print();
        ok = 2;
    }

    Py_Finalize();
    return ok;
} 
