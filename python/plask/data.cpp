#include "python_globals.h"
#include "python_provider.h"
#include <numpy/arrayobject.h>

#include <plask/mesh/mesh.h>
#include <plask/data.h>

namespace plask { namespace python {


template <typename T, int dim>
const typename DataVector<T>::const_iterator DataVectorWrap_begin(const DataVectorWrap<T,dim>& self) { return self.begin(); }

template <typename T, int dim>
const typename DataVector<T>::const_iterator DataVectorWrap_end(const DataVectorWrap<T,dim>& self) { return self.end(); }

template <typename T, int dim>
T DataVectorWrap_getitem(const DataVectorWrap<T,dim>& self, std::ptrdiff_t i) {
    if (i < 0) i = self.size() - i;
    if (i < 0 || i >= self.size()) throw IndexError("index out of range");
    return self[i];
}

template <typename T> inline static int get_typenum();
template <> int get_typenum<double>() { return NPY_DOUBLE; }
template <> int get_typenum<dcomplex>() { return NPY_CDOUBLE; }
template <> int get_typenum<Vec<2,double>>() { return NPY_DOUBLE; }
template <> int get_typenum<Vec<2,dcomplex>>() { return NPY_CDOUBLE; }
template <> int get_typenum<Vec<3,double>>() { return NPY_DOUBLE; }
template <> int get_typenum<Vec<3,dcomplex>>() { return NPY_CDOUBLE; }

template <typename T> inline static std::vector<npy_intp> get_dims(size_t n) { std::vector<npy_intp> dims(1); dims[0] = n; return dims; }
template <> inline std::vector<npy_intp> get_dims<Vec<2,double>>(size_t n) { std::vector<npy_intp> dims(2); dims[0] = n; dims[1] = 2; return dims; }
template <> inline std::vector<npy_intp> get_dims<Vec<2,dcomplex>>(size_t n) { std::vector<npy_intp> dims(2); dims[0] = n; dims[1] = 2; return dims; }
template <> inline std::vector<npy_intp> get_dims<Vec<3,double>>(size_t n) { std::vector<npy_intp> dims(2); dims[0] = n; dims[1] = 3; return dims; }
template <> inline std::vector<npy_intp> get_dims<Vec<3,dcomplex>>(size_t n) { std::vector<npy_intp> dims(2); dims[0] = n; dims[1] = 3; return dims; }



template <typename T, int dim>
py::object DataVectorWrap_getslice(const DataVectorWrap<T,dim>& self, std::ptrdiff_t from, std::ptrdiff_t to) {
    if (from < 0) from = self.size() - from;
    if (to < 0) to = self.size() - to;
    if (from < 0) from = 0;
    if (to > self.size()) to = self.size();

    // TODO return array
    std::vector<npy_intp> dims = get_dims<T>(to-from);
    PyObject* arr = PyArray_SimpleNew(dims.size(), &dims.front(), get_typenum<T>());
    T* arr_data = (T*)PyArray_DATA(arr);
    for (auto i = self.begin()+from; i < self.begin()+to; ++i, ++arr_data)
        *arr_data = *i;
    return py::object(py::handle<>(arr));
}

template <typename T, int dim>
bool DataVectorWrap_contains(const DataVectorWrap<T,dim>& self, const T& key) { return std::find(self.begin(), self.end(), key) != self.end(); }


template <typename T, int dim>
py::object DataVectorWrap__array(py::object oself) {
    const DataVectorWrap<T,dim>* self = py::extract<const DataVectorWrap<T,dim>*>(oself);
    std::vector<npy_intp> dims = get_dims<T>(self->size());
    PyObject* arr = PyArray_SimpleNewFromData(dims.size(), &dims.front(), get_typenum<T>(), (void*)self->data());
    if (arr == nullptr) throw plask::CriticalException("cannot create array from data");
    py::incref(oself.ptr()); PyArray_BASE(arr) = oself.ptr(); // Make sure the data vector stays alive as long as the array
    return py::object(py::handle<>(arr));
}



template <typename T, int dim>
void register_data_vector() {

    py::class_<DataVectorWrap<T,dim>, shared_ptr<DataVectorWrap<T,dim>>>("DataVector", "Data returned by field providers", py::no_init)
        .def_readonly("mesh", &DataVectorWrap<T,dim>::mesh)
        .def("__len__", &DataVectorWrap<T,dim>::size)
        .def("__getitem__", &DataVectorWrap_getitem<T,dim>)
        .def("__getslice__", &DataVectorWrap_getslice<T,dim>)
        .def("__contains__", &DataVectorWrap_contains<T,dim>)
        .def("__iter__", py::range(&DataVectorWrap_begin<T,dim>, &DataVectorWrap_end<T,dim>))
        .def("__array__", &DataVectorWrap__array<T,dim>)
    ;

}

static inline bool plask_import_array() {
    import_array1(false);
    return true;
}

void register_data_vectors() {
    // Initialize numpy
    if (!plask_import_array()) throw(py::error_already_set());

    register_data_vector<double, 2>();
    register_data_vector<dcomplex, 2>();
}

}} // namespace plask::python

#include <plask/data.h>

