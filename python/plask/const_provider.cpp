#include "python_provider.h"

namespace plask { namespace python {

PythonStepProfile::Place::Place(py::object src) {
    GeometryObject* obj;
    PathHints hnts;
    try {
        obj = py::extract<GeometryObject*>(src);
    } catch (py::error_already_set) {
        try {
            PyErr_Clear();
            if (py::len(src) != 2) throw py::error_already_set();
            obj = py::extract<GeometryObject*>(src[0]);
            hnts = py::extract<PathHints>(src[1]);
        } catch (py::error_already_set) {
            throw TypeError("Key must be either of type geometry.GeometryObject or (geometry.GeometryObject, geometry.PathHints)");
        }
    }
    object = obj->shared_from_this();
    hints = hnts;
}


py::object PythonStepProfile::__getitem__(py::object key) {
    Place place(key);
    auto found = std::find(places.begin(), places.end(), place);
    if (found != places.end()) return values[found-places.begin()];
    return custom_default_value;
}


void PythonStepProfile::__setitem__(py::object key, py::object value) {
    Place place(key);
    auto found = std::find(places.begin(), places.end(), place);
    if (found != places.end()) values[found-places.begin()] = value;
    else {
        places.push_front(place);
        values.push_front(value);
    }
    fireChanged();
}


void PythonStepProfile::__delitem__(py::object key) {
    auto found = std::find(places.begin(), places.end(), Place(key));
    if (found != places.end()) { places.erase(found); values.erase(values.begin() + (found-places.begin())); fireChanged(); }
}


void PythonStepProfile::clear() {
    places.clear();
    values.clear();
    fireChanged();
}


py::list PythonStepProfile::keys() const {
    py::list result;
    for (auto place: places) result.insert(0, place);
    return result;
}


py::list PythonStepProfile::pyvalues() const {
    py::list result;
    for (auto value: values) result.insert(0, value);
    return result;
}

struct PythonStepProfile_Place_to_python
{
    static PyObject* convert(PythonStepProfile::Place const& item) {
        return py::incref(py::make_tuple(item.object.lock(), item.hints).ptr());
    }
};



void register_step_profile()
{
    py::class_<PythonStepProfile, shared_ptr<PythonStepProfile>, boost::noncopyable>("StepProfileProvider",
        "General step profile provider\n\n"
        "StepProfileProvider(geometry[, default_value])\n    Create step profile for specified geometry and optional default_value\n\n",
        py::init<const Geometry&, py::object>((py::arg("geometry"), py::arg("default_value")=py::object())))
        .def("__getitem__", &PythonStepProfile::__getitem__)
        .def("__setitem__", &PythonStepProfile::__setitem__)
        .def("__delitem__", &PythonStepProfile::__delitem__)
        .def("__len__", &PythonStepProfile::size)
        .def("clear", &PythonStepProfile::clear, "Clear values for all places")
        .def("keys", &PythonStepProfile::keys, "Show list of defined places")
        .def("values", &PythonStepProfile::pyvalues, "Show list of defined values")
        ;

    py::to_python_converter<PythonStepProfile::Place, PythonStepProfile_Place_to_python>();
}

}} // namespace plask::python
