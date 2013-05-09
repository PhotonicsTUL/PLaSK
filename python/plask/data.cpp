#include "python_globals.h"
#include "python_provider.h"
#include "python_numpy.h"

#include <plask/mesh/mesh.h>
#include <plask/mesh/rectilinear.h>
#include <plask/mesh/regular.h>
#include <plask/data.h>
#include <plask/vec.h>

namespace plask { namespace python {


/*
 * Some helper functions for getting information on rectangular meshes
 */
namespace detail {

    template <typename T> constexpr inline static npy_intp type_dim() { return 1; }
    template <> constexpr inline npy_intp type_dim<Vec<2,double>>() { return 2; }
    template <> constexpr inline npy_intp type_dim<Vec<2,dcomplex>>() { return 2; }
    template <> constexpr inline npy_intp type_dim<Vec<3,double>>() { return 3; }
    template <> constexpr inline npy_intp type_dim<Vec<3,dcomplex>>() { return 3; }
    template <> constexpr inline npy_intp type_dim<const Vec<2,double>>() { return 2; }
    template <> constexpr inline npy_intp type_dim<const Vec<2,dcomplex>>() { return 2; }
    template <> constexpr inline npy_intp type_dim<const Vec<3,double>>() { return 3; }
    template <> constexpr inline npy_intp type_dim<const Vec<3,dcomplex>>() { return 3; }
    template <> constexpr inline npy_intp type_dim<const Tensor2<double>>() { return 2; }
    template <> constexpr inline npy_intp type_dim<const Tensor3<dcomplex>>() { return 5; }


    inline static std::vector<npy_intp> mesh_dims(const RectilinearMesh2D& mesh) { return { mesh.axis0.size(), mesh.axis1.size() }; }
    inline static std::vector<npy_intp> mesh_dims(const RectilinearMesh3D& mesh) { return { mesh.axis0.size(), mesh.axis1.size(), mesh.axis2.size() }; }
    inline static std::vector<npy_intp> mesh_dims(const RegularMesh2D& mesh) { return { mesh.axis0.size(), mesh.axis1.size() }; }
    inline static std::vector<npy_intp> mesh_dims(const RegularMesh3D& mesh) { return { mesh.axis0.size(), mesh.axis1.size(), mesh.axis2.size() }; }


    template <typename T, typename Mesh1D>
    inline static std::vector<npy_intp> mesh_strides(const RectangularMesh<2,Mesh1D>& mesh, size_t nd) {
        std::vector<npy_intp> strides(nd);
        strides.back() = sizeof(T) / type_dim<T>();
        if (mesh.getIterationOrder() == RectangularMesh<2,Mesh1D>::NORMAL_ORDER) {
            strides[0] = sizeof(T);
            strides[1] = mesh.axis0.size() * sizeof(T);
        } else {
            strides[0] = mesh.axis1.size() * sizeof(T);
            strides[1] = sizeof(T);
        }
        return strides;
    }

    #define ITERATION_ORDER_STRIDE_CASE_RECTILINEAR(MeshT, first, second, third) \
        case MeshT::ORDER_##first##second##third: \
            strides[first] = mesh.axis##second.size() * mesh.axis##third.size() * sizeof(T); \
            strides[second] = mesh.axis##third.size() * sizeof(T); \
            strides[third] = sizeof(T); \
            break;

    template <typename T, typename Mesh1D>
    inline static std::vector<npy_intp> mesh_strides(const RectangularMesh<3,Mesh1D>& mesh, size_t nd) {
        std::vector<npy_intp> strides(nd, sizeof(T)/type_dim<T>());
        typedef RectangularMesh<3,Mesh1D> Mesh3D;
        switch (mesh.getIterationOrder()) {
            ITERATION_ORDER_STRIDE_CASE_RECTILINEAR(Mesh3D, 0,1,2)
            ITERATION_ORDER_STRIDE_CASE_RECTILINEAR(Mesh3D, 0,2,1)
            ITERATION_ORDER_STRIDE_CASE_RECTILINEAR(Mesh3D, 1,0,2)
            ITERATION_ORDER_STRIDE_CASE_RECTILINEAR(Mesh3D, 1,2,0)
            ITERATION_ORDER_STRIDE_CASE_RECTILINEAR(Mesh3D, 2,0,1)
            ITERATION_ORDER_STRIDE_CASE_RECTILINEAR(Mesh3D, 2,1,0)
        }
        return strides;
    }

    #undef ITERATION_ORDER_STRIDE_CASE_RECTILINEAR

} // namespace  detail




template <typename T, int dim>
static const typename DataVector<T>::const_iterator DataVectorWrap_begin(const DataVectorWrap<T,dim>& self) { return self.begin(); }

template <typename T, int dim>
static const typename DataVector<T>::const_iterator DataVectorWrap_end(const DataVectorWrap<T,dim>& self) { return self.end(); }

template <typename T, int dim>
static T DataVectorWrap_getitem(const DataVectorWrap<T,dim>& self, std::ptrdiff_t i) {
    if (i < 0) i += self.size();
    if (i < 0 || std::size_t(i) >= self.size()) throw IndexError("index out of range");
    return self[i];
}




template <typename T, int dim>
static py::object DataVectorWrap_getslice(const DataVectorWrap<T,dim>& self, std::ptrdiff_t from, std::ptrdiff_t to) {
    if (from < 0) from = self.size() - from;
    if (to < 0) to = self.size() - to;
    if (from < 0) from = 0;
    if (std::size_t(to) > self.size()) to = self.size();

    npy_intp dims[] = { to-from, detail::type_dim<T>() };
    PyObject* arr = PyArray_SimpleNew((dims[1]!=1)? 2 : 1, dims, detail::typenum<T>());
    typename std::remove_const<T>::type* arr_data = static_cast<typename std::remove_const<T>::type*>(PyArray_DATA(arr));
    for (auto i = self.begin()+from; i < self.begin()+to; ++i, ++arr_data)
        *arr_data = *i;
    return py::object(py::handle<>(arr));
}


template <typename T, int dim>
static bool DataVectorWrap_contains(const DataVectorWrap<T,dim>& self, const T& key) { return std::find(self.begin(), self.end(), key) != self.end(); }


template <typename T, int dim>
py::handle<> DataVector_dtype() {
    return detail::dtype<typename std::remove_const<T>::type>();
}



template <typename T, int dim>
static py::object DataVectorWrap__array__(py::object oself, py::object dtype) {

    const DataVectorWrap<T,dim>* self = py::extract<const DataVectorWrap<T,dim>*>(oself);

    if (self->mesh_changed) throw Exception("Cannot create array, mesh changed since data retrieval");

    npy_intp dims[] = { self->mesh->size() * detail::type_dim<T>() };

    PyObject* arr = PyArray_SimpleNewFromData(1, dims, detail::typenum<T>(), (void*)self->data());
    if (arr == nullptr) throw plask::CriticalException("Cannot create array from data");

    confirm_array<T>(arr, oself, dtype);

    return py::object(py::handle<>(arr));
}



template <typename T, typename MeshT, int dim>
static PyObject* DataVectorWrap_ArrayImpl(const DataVectorWrap<T,dim>* self) {
    shared_ptr<MeshT> mesh = dynamic_pointer_cast<MeshT>(self->mesh);
    if (!mesh) return nullptr;

    std::vector<npy_intp> dims = detail::mesh_dims(*mesh);
    if (detail::type_dim<T>() != 1) dims.push_back(detail::type_dim<T>());

    PyObject* arr = PyArray_New(&PyArray_Type,
                                dims.size(),
                                & dims.front(),
                                detail::typenum<T>(),
                                & detail::mesh_strides<T>(*mesh, dims.size()).front(),
                                (void*)self->data(),
                                0,
                                0,
                                NULL);
    if (arr == nullptr) throw plask::CriticalException("Cannot create array from data");
    return arr;
}

template <typename T, int dim>
static py::object DataVectorWrap_Array(py::object oself) {
    const DataVectorWrap<T,dim>* self = py::extract<const DataVectorWrap<T,dim>*>(oself);

    if (self->mesh_changed) throw Exception("Cannot create array, mesh changed since data retrieval");

    PyObject* arr = DataVectorWrap_ArrayImpl<T, RectilinearMesh2D>(self);
    if (!arr) arr = DataVectorWrap_ArrayImpl<T, RectilinearMesh3D>(self);
    if (!arr) arr = DataVectorWrap_ArrayImpl<T, RegularMesh2D>(self);
    if (!arr) arr = DataVectorWrap_ArrayImpl<T, RegularMesh3D>(self);

    if (arr == nullptr) throw TypeError("Cannot create array for data on this mesh type (possible only for %1%)",
                                        (dim == 2)? "mesh.RegularMesh2D or mesh.RectilinearMesh2D" : "mesh.RegularMesh3D or mesh.RectilinearMesh3D");

    py::incref(oself.ptr());
    PyArray_BASE(arr) = oself.ptr(); // Make sure the data vector stays alive as long as the array
    // confirm_array<T>(arr, oself, dtype);

    return py::object(py::handle<>(arr));
}





namespace detail {

    template <typename T, typename MeshT>
    static size_t checkMeshAndArray(PyArrayObject* arr, const MeshT& mesh) {
        auto mesh_dims = detail::mesh_dims(mesh);
        if (detail::type_dim<T>() != 1)  mesh_dims.push_back(detail::type_dim<T>());
        size_t nd = mesh_dims.size();

        if ((size_t)PyArray_NDIM(arr) != nd) throw ValueError("Provided array must have either 1 or %1% dimensions", MeshT::DIM);

        for (size_t i = 0; i != nd; ++i)
            if (mesh_dims[i] != PyArray_DIMS(arr)[i])
                throw ValueError("Dimension %1% for the array (%3%) does not match with the mesh (%2%)", i, mesh_dims[i], PyArray_DIMS(arr)[i]);

        auto mesh_strides = detail::mesh_strides<T>(mesh, nd);
        for (size_t i = 0; i != nd; ++i)
            if (mesh_strides[i] != PyArray_STRIDES(arr)[i])
                throw ValueError("Stride %1% for the array do not correspond to the current mesh ordering", i);

        return mesh.size();
    }

    struct NumpyDataDeleter {
        PyArrayObject* arr;
        NumpyDataDeleter(PyArrayObject* arr) : arr(arr) { Py_XINCREF(arr); }
        void operator()(void*) { Py_XDECREF(arr); }
    };

    template <typename T, int dim>
    static py::object makeDataVectorImpl(PyArrayObject* arr, shared_ptr<MeshD<dim>> mesh) {

        size_t size;

        auto regular = dynamic_pointer_cast<RectangularMesh<dim,RegularMesh1D>>(mesh);
        auto rectilinear = dynamic_pointer_cast<RectangularMesh<dim,RectilinearMesh1D>>(mesh);

        if (PyArray_NDIM(arr) != 1) {

            if (regular) size = checkMeshAndArray<T, RectangularMesh<dim,RegularMesh1D>>(arr, *regular);
            else if (rectilinear) size = checkMeshAndArray<T, RectangularMesh<dim,RectilinearMesh1D>>(arr, *rectilinear);
            else throw TypeError("For this mesh type only one-dimensional array is allowed");

        } else
            size = PyArray_DIMS(arr)[0] / type_dim<T>();

        if (size != mesh->size()) throw ValueError("Sizes of data (%1%) and mesh (%2%) do not match", size, mesh->size());

        auto data = make_shared<DataVectorWrap<const T,dim>>(
            DataVector<const T>((const T*)PyArray_DATA(arr), size, NumpyDataDeleter(arr)),
            mesh);

        return py::object(data);
    }

    template <typename T, int dim>
    static py::object makeDataVector(PyArrayObject* arr, shared_ptr<MeshD<dim>> mesh) {
        size_t ndim = PyArray_NDIM(arr);
        size_t last_dim = PyArray_DIMS(arr)[ndim-1];
        if (ndim == dim+1) {
            if (last_dim == 2) return makeDataVectorImpl<Vec<2,T>, dim>(arr, mesh);
            else if (last_dim == 3) return makeDataVectorImpl<Vec<2,T>, dim>(arr, mesh);
        } else if (ndim == 1) {
            if (last_dim == 2 * mesh->size()) return makeDataVectorImpl<Vec<2,T>, dim>(arr, mesh);
            else if (last_dim == 3 * mesh->size()) return makeDataVectorImpl<Vec<3,T>, dim>(arr, mesh);
        }
        return makeDataVectorImpl<double, dim>(arr, mesh);
    }

} // namespace  detail

py::object Data(PyObject* obj, py::object omesh) {
    if (!PyArray_Check(obj)) throw TypeError("data needs to be array object");
    PyArrayObject* arr = (PyArrayObject*)obj;

    try {
        shared_ptr<MeshD<2>> mesh = py::extract<shared_ptr<MeshD<2>>>(omesh);

        switch (PyArray_TYPE(arr)) {
            case NPY_DOUBLE: return detail::makeDataVector<double,2>(arr, mesh);
            case NPY_CDOUBLE: return detail::makeDataVector<dcomplex,2>(arr, mesh);
            default: throw TypeError("Array has wrong dtype (only float and complex allowed)");
        }

    } catch (py::error_already_set) { PyErr_Clear(); try {
        shared_ptr<MeshD<3>> mesh = py::extract<shared_ptr<MeshD<3>>>(omesh);

        switch (PyArray_TYPE(arr)) {
            case NPY_DOUBLE: return detail::makeDataVector<double,3>(arr, mesh);
            case NPY_CDOUBLE: return detail::makeDataVector<dcomplex,3>(arr, mesh);
            default: throw TypeError("Array has wrong dtype (only float and complex allowed)");
        }

    } catch (py::error_already_set) {
        throw TypeError("mesh must be a proper mesh object");
    }}

    return py::object();
}


template <typename T, int dim>
void register_data_vector() {

    py::class_<DataVectorWrap<const T,dim>, shared_ptr<DataVectorWrap<const T,dim>>>("Data", "Data returned by field providers", py::no_init)
        .def_readonly("mesh", &DataVectorWrap<const T,dim>::mesh)
        .def("__len__", &DataVectorWrap<const T,dim>::size)
        .def("__getitem__", &DataVectorWrap_getitem<const T,dim>)
        .def("__getslice__", &DataVectorWrap_getslice<const T,dim>)
        .def("__contains__", &DataVectorWrap_contains<const T,dim>)
        .def("__iter__", py::range(&DataVectorWrap_begin<const T,dim>, &DataVectorWrap_end<const T,dim>))
        .def("__array__", &DataVectorWrap__array__<const T,dim>, py::arg("dtype")=py::object())
        .add_property("array", &DataVectorWrap_Array<const T,dim>, "Array formatted by the mesh")
        .add_static_property("dtype", &DataVector_dtype<const T,dim>, "Type of the held values")
    ;
}

template <int dim>
static inline void register_data_vectors_d() {
    register_data_vector<double, dim>();
    register_data_vector<dcomplex, dim>();
    register_data_vector<Vec<2,double>, dim>();
    register_data_vector<Vec<2,dcomplex>, dim>();
    register_data_vector<Vec<3,double>, dim>();
    register_data_vector<Vec<3,dcomplex>, dim>();
    register_data_vector<Tensor2<double>, dim>();
    register_data_vector<Tensor3<dcomplex>, dim>();
}

void register_data_vectors() {
    // Initialize numpy
    if (!plask_import_array()) throw(py::error_already_set());

    register_data_vectors_d<2>();
    register_data_vectors_d<3>();

    py::def("Data", &Data, (py::arg("array"), "mesh"), "Create new data from array and mesh");
}

}} // namespace plask::python

#include <plask/data.h>

