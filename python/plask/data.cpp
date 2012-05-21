#include "python_globals.h"
#include "python_provider.h"

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

template <typename T, int dim>
py::list DataVectorWrap_getslice(const DataVectorWrap<T,dim>& self, std::ptrdiff_t from, std::ptrdiff_t to) {
    if (from < 0) from = self.size() - from;
    if (to < 0) to = self.size() - to;
    if (from < 0) from = 0;
    if (to > self.size()) to = self.size();

    // TODO return array
    py::list result;
    for (auto i = self.begin()+from; i < self.begin()+to; ++i)
        result.append(*i);
    return result;
}

template <typename T, int dim>
bool DataVectorWrap_contains(const DataVectorWrap<T,dim>& self, const T& key) { return std::find(self.begin(), self.end(), key) != self.end(); }


template <typename T, int dim>
py::object DataVectorWrap__array(const DataVectorWrap<T,dim>& self) {
        // TODO
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

void register_data_vectors() {
    register_data_vector<double, 2>();
    register_data_vector<dcomplex, 2>();
}

}} // namespace plask::python

#include <plask/data.h>

