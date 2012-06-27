#include "python_globals.h"
#include "python_provider.h"
#include <numpy/arrayobject.h>

#include <plask/mesh/mesh.h>
#include <plask/mesh/rectilinear.h>
#include <plask/data.h>
#include <plask/vec.h>

namespace plask { namespace python {


template <typename T, int dim>
static const typename DataVector<T>::const_iterator DataVectorWrap_begin(const DataVectorWrap<T,dim>& self) { return self.begin(); }

template <typename T, int dim>
static const typename DataVector<T>::const_iterator DataVectorWrap_end(const DataVectorWrap<T,dim>& self) { return self.end(); }

template <typename T, int dim>
static T DataVectorWrap_getitem(const DataVectorWrap<T,dim>& self, std::ptrdiff_t i) {
    if (i < 0) i = self.size() - i;
    if (i < 0 || std::size_t(i) >= self.size()) throw IndexError("index out of range");
    return self[i];
}

template <typename T> constexpr inline static npy_intp get_dim() { return 1; }
template <> constexpr inline npy_intp get_dim<Vec<2,double>>() { return 2; }
template <> constexpr inline npy_intp get_dim<Vec<2,dcomplex>>() { return 2; }
template <> constexpr inline npy_intp get_dim<Vec<3,double>>() { return 3; }
template <> constexpr inline npy_intp get_dim<Vec<3,dcomplex>>() { return 3; }




template <typename T, int dim>
static py::object DataVectorWrap_getslice(const DataVectorWrap<T,dim>& self, std::ptrdiff_t from, std::ptrdiff_t to) {
    if (from < 0) from = self.size() - from;
    if (to < 0) to = self.size() - to;
    if (from < 0) from = 0;
    if (std::size_t(to) > self.size()) to = self.size();

    npy_intp dims[] = { to-from, get_dim<T>() };
    PyObject* arr = PyArray_SimpleNew((dims[1]!=1)? 2 : 1, dims, detail::get_typenum<T>());
    T* arr_data = (T*)PyArray_DATA(arr);
    for (auto i = self.begin()+from; i < self.begin()+to; ++i, ++arr_data)
        *arr_data = *i;
    return py::object(py::handle<>(arr));
}

template <typename T, int dim>
static bool DataVectorWrap_contains(const DataVectorWrap<T,dim>& self, const T& key) { return std::find(self.begin(), self.end(), key) != self.end(); }





inline std::vector<npy_intp> get_meshdims(const RectilinearMesh2D& mesh) { return { mesh.c1.size(), mesh.c0.size() }; }
inline std::vector<npy_intp> get_meshdims(const RectilinearMesh3D& mesh) { return { mesh.c2.size(), mesh.c1.size(), mesh.c0.size() }; }
// inline std::vector<npy_intp> get_meshdims(const RegularMesh2D& mesh) { return { mesh.c1.size(), mesh.c0.size() }; }
// inline std::vector<npy_intp> get_meshdims(const RegularMesh3D& mesh) { return { mesh.c2.size(), mesh.c1.size(), mesh.c0.size() }; }


template <typename T>
inline std::vector<npy_intp> get_meshstrides(const RectilinearMesh2D& mesh, size_t nd) {
    std::vector<npy_intp> strides(nd, sizeof(T)/get_dim<T>());
    if (mesh.getIterationOrder() == RectilinearMesh2D::NORMAL_ORDER) {
        strides[0] = mesh.c0.size() * sizeof(T);
        strides[1] = sizeof(T);
    } else {
        strides[0] = sizeof(T);
        strides[1] = mesh.c1.size() * sizeof(T);
    }
    return strides;
}

#define ITERATION_ORDER_STRIDE_CASE_RECTILINEAR(first, second, third) \
    case RectilinearMesh3D::ORDER_##first##second##third: \
        strides[2-first] = mesh.c##second.size() * mesh.c##third.size() * sizeof(T); \
        strides[2-second] = mesh.c##third.size() * sizeof(T); \
        strides[2-third] = sizeof(T); \
        break;
template <typename T>
inline std::vector<npy_intp> get_meshstrides(const RectilinearMesh3D& mesh, size_t nd) {
    std::vector<npy_intp> strides(nd, sizeof(T)/get_dim<T>());
    switch (mesh.getIterationOrder()) {
        ITERATION_ORDER_STRIDE_CASE_RECTILINEAR(0,1,2)
        ITERATION_ORDER_STRIDE_CASE_RECTILINEAR(0,2,1)
        ITERATION_ORDER_STRIDE_CASE_RECTILINEAR(1,0,2)
        ITERATION_ORDER_STRIDE_CASE_RECTILINEAR(1,2,0)
        ITERATION_ORDER_STRIDE_CASE_RECTILINEAR(2,0,1)
        ITERATION_ORDER_STRIDE_CASE_RECTILINEAR(2,1,0)
    }
    return strides;
}


template <typename T, int dim, typename MeshT>
static PyObject* DataVectorWrap_ArrayImpl(const DataVectorWrap<T,dim>* self) {
    shared_ptr<MeshT> mesh = dynamic_pointer_cast<MeshT>(self->mesh);
    if (!mesh) return nullptr;

    std::vector<npy_intp> dims = get_meshdims(*mesh);
    if (get_dim<T>() != 1) dims.push_back(get_dim<T>());

    PyObject* arr = PyArray_New(&PyArray_Type,
                                dims.size(),
                                & dims.front(),
                                detail::get_typenum<T>(),
                                & get_meshstrides<T>(*mesh, dims.size()).front(),
                                (void*)self->data(),
                                0,
                                0,
                                NULL);

    if (arr == nullptr) throw plask::CriticalException("cannot create array from data");
    return arr;
}


template <typename T, int dim>
static py::object DataVectorWrap_Array(py::object oself) {
    const DataVectorWrap<T,dim>* self = py::extract<const DataVectorWrap<T,dim>*>(oself);

    if (self->mesh_changed) throw Exception("cannot create array, mesh changed since data retrieval");

    PyObject* arr = DataVectorWrap_ArrayImpl<T, dim, RectilinearMesh2D>(self);
    if (!arr) arr = DataVectorWrap_ArrayImpl<T, dim, RectilinearMesh3D>(self);

    if (arr == nullptr) throw TypeError("cannot create array for data on this mesh type");

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
        .add_property("array", &DataVectorWrap_Array<T,dim>)
    ;

    // TODO Podłączenie do receivera
    // serializacja?

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
    register_data_vector< Vec<2,double>, 2 >();
    register_data_vector< Vec<2,dcomplex>, 2 >();
    register_data_vector< Vec<3,double>, 2 >();
    register_data_vector< Vec<3,dcomplex>, 2 >();

    register_data_vector<double, 3>();
    register_data_vector<dcomplex, 3>();
    register_data_vector< Vec<3,double>, 3 >();
    register_data_vector< Vec<3,dcomplex>, 3 >();
}

}} // namespace plask::python

#include <plask/data.h>

