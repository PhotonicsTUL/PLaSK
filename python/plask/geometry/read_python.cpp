#include <plask/plask.hpp>
#include "../python_globals.hpp"


#define PLASK_GEOMETRY_PYTHON_TAG "python"
#define RETURN_VARIABLE "__object__"

namespace plask { namespace python {

extern AxisNames current_axes;
extern PLASK_PYTHON_API py::dict* pyXplGlobals;

namespace detail {
    struct SetPythonAxes {
        AxisNames saved;
        SetPythonAxes(GeometryReader& reader): saved(current_axes) {
            current_axes = reader.getAxisNames();
        }
        ~SetPythonAxes() {
            current_axes = saved;
        }
    };
}

shared_ptr<GeometryObject> read_python(GeometryReader& reader) {
    size_t linenp = reader.source.getLineNr();
    PyCodeObject* code = compilePythonFromXml(reader.source);
    detail::SetPythonAxes setPythonAxes(reader);

    py::dict locals;

    PyObject* result = PyEval_EvalCode((PyObject*)code, pyXplGlobals->ptr(), locals.ptr());
    if (!result) throw py::error_already_set();

    if (result == Py_None) {
        Py_DECREF(result);
        if (locals.has_key(RETURN_VARIABLE)) {
            result = PyDict_GetItemString(locals.ptr(), RETURN_VARIABLE);
            Py_INCREF(result);
        } else {
            throw XMLException(reader.source, "No geometry item defined");
        }
    }
    py::handle<> hres(result);

    return py::extract<shared_ptr<GeometryObject>>(result);
}

static GeometryReader::RegisterObjectReader python_reader(PLASK_GEOMETRY_PYTHON_TAG, read_python);

}} // namespace plask::python
