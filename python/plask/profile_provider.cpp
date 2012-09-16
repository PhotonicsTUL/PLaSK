#include "python_provider.h"

namespace plask { namespace python {

PythonProfile::Place::Place(py::object src) {
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


py::object PythonProfile::__getitem__(py::object key) {
    Place place(key);
    auto found = std::find(places.begin(), places.end(), place);
    if (found != places.end()) return values[found-places.begin()];
    return custom_default_value;
}


void PythonProfile::__setitem__(py::object key, py::object value) {
    Place place(key);
    auto found = std::find(places.begin(), places.end(), place);
    if (found != places.end()) values[found-places.begin()] = value;
    else {
        places.push_front(place);
        values.push_front(value);
    }
    fireChanged();
}


void PythonProfile::__delitem__(py::object key) {
    auto found = std::find(places.begin(), places.end(), Place(key));
    if (found != places.end()) { places.erase(found); values.erase(values.begin() + (found-places.begin())); fireChanged(); }
}


void PythonProfile::clear() {
    places.clear();
    values.clear();
    fireChanged();
}


py::list PythonProfile::keys() const {
    py::list result;
    for (auto place: places) result.insert(0, place);
    return result;
}


py::list PythonProfile::pyvalues() const {
    py::list result;
    for (auto value: values) result.insert(0, value);
    return result;
}

struct PythonProfile_Place_to_python
{
    static PyObject* convert(PythonProfile::Place const& item) {
        return py::incref(py::make_tuple(item.object.lock(), item.hints).ptr());
    }
};



void register_step_profile()
{
    py::class_<PythonProfile, shared_ptr<PythonProfile>, boost::noncopyable>("StepProfile",
        "General step profile provider\n\n"
        "StepProfile(geometry[, default_value])\n    Create step profile for specified geometry and optional default_value\n\n",
        py::init<const Geometry&, py::object>((py::arg("geometry"), py::arg("default_value")=py::object())))
        .def("__getitem__", &PythonProfile::__getitem__)
        .def("__setitem__", &PythonProfile::__setitem__)
        .def("__delitem__", &PythonProfile::__delitem__)
        .def("__len__", &PythonProfile::size)
        .def("clear", &PythonProfile::clear, "Clear values for all places")
        .def("keys", &PythonProfile::keys, "Show list of defined places")
        .def("values", &PythonProfile::pyvalues, "Show list of defined values")
        ;

    py::to_python_converter<PythonProfile::Place, PythonProfile_Place_to_python>();
}

}} // namespace plask::python
