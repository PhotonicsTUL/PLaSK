#include "python_globals.h"
#include "python_provider.h"
#include <numpy/arrayobject.h>

#include <plask/mesh/mesh.h>
#include <plask/mesh/rectilinear.h>
#include <plask/mesh/regular.h>
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

extern py::class_<Vec<2,double>> vector2fClass;
extern py::class_<Vec<2,dcomplex>> vector2cClass;
extern py::class_<Vec<3,double>> vector3fClass;
extern py::class_<Vec<3,dcomplex>> vector3cClass;

// dtype
inline static py::handle<> dtype(const double&) { return py::handle<>(py::borrowed<>(reinterpret_cast<PyObject*>(&PyFloat_Type))); }
inline static py::handle<> dtype(const Vec<2,double>&) { return py::handle<>(py::borrowed<>(reinterpret_cast<PyObject*>(vector2fClass.ptr()))); }
inline static py::handle<> dtype(const Vec<3,double>&) { return py::handle<>(py::borrowed<>(reinterpret_cast<PyObject*>(vector2cClass.ptr()))); }
inline static py::handle<> dtype(const dcomplex&) { return py::handle<>(py::borrowed<>(reinterpret_cast<PyObject*>(&PyComplex_Type))); }
inline static py::handle<> dtype(const Vec<2,dcomplex>&) { return py::handle<>(py::borrowed<>(reinterpret_cast<PyObject*>(vector3fClass.ptr()))); }
inline static py::handle<> dtype(const Vec<3,dcomplex>&) { return py::handle<>(py::borrowed<>(reinterpret_cast<PyObject*>(vector3cClass.ptr()))); }

template <typename T, int dim>
py::handle<> DataVector_dtype() {
    return dtype(T());
}


inline std::vector<npy_intp> get_meshdims(const RectilinearMesh2D& mesh) { return { mesh.c1.size(), mesh.c0.size() }; }
inline std::vector<npy_intp> get_meshdims(const RectilinearMesh3D& mesh) { return { mesh.c2.size(), mesh.c1.size(), mesh.c0.size() }; }
inline std::vector<npy_intp> get_meshdims(const RegularMesh2D& mesh) { return { mesh.c1.size(), mesh.c0.size() }; }
inline std::vector<npy_intp> get_meshdims(const RegularMesh3D& mesh) { return { mesh.c2.size(), mesh.c1.size(), mesh.c0.size() }; }


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

template <typename T>
inline std::vector<npy_intp> get_meshstrides(const RegularMesh2D& mesh, size_t nd) {
    std::vector<npy_intp> strides(nd, sizeof(T)/get_dim<T>());
    if (mesh.getIterationOrder() == RegularMesh2D::NORMAL_ORDER) {
        strides[0] = mesh.c0.size() * sizeof(T);
        strides[1] = sizeof(T);
    } else {
        strides[0] = sizeof(T);
        strides[1] = mesh.c1.size() * sizeof(T);
    }
    return strides;
}

#define ITERATION_ORDER_STRIDE_CASE_RECTILINEAR(MeshT, first, second, third) \
    case MeshT::ORDER_##first##second##third: \
        strides[2-first] = mesh.c##second.size() * mesh.c##third.size() * sizeof(T); \
        strides[2-second] = mesh.c##third.size() * sizeof(T); \
        strides[2-third] = sizeof(T); \
        break;

template <typename T>
inline std::vector<npy_intp> get_meshstrides(const RectilinearMesh3D& mesh, size_t nd) {
    std::vector<npy_intp> strides(nd, sizeof(T)/get_dim<T>());
    switch (mesh.getIterationOrder()) {
        ITERATION_ORDER_STRIDE_CASE_RECTILINEAR(RectilinearMesh3D, 0,1,2)
        ITERATION_ORDER_STRIDE_CASE_RECTILINEAR(RectilinearMesh3D, 0,2,1)
        ITERATION_ORDER_STRIDE_CASE_RECTILINEAR(RectilinearMesh3D, 1,0,2)
        ITERATION_ORDER_STRIDE_CASE_RECTILINEAR(RectilinearMesh3D, 1,2,0)
        ITERATION_ORDER_STRIDE_CASE_RECTILINEAR(RectilinearMesh3D, 2,0,1)
        ITERATION_ORDER_STRIDE_CASE_RECTILINEAR(RectilinearMesh3D, 2,1,0)
    }
    return strides;
}

template <typename T>
inline std::vector<npy_intp> get_meshstrides(const RegularMesh3D& mesh, size_t nd) {
    std::vector<npy_intp> strides(nd, sizeof(T)/get_dim<T>());
    switch (mesh.getIterationOrder()) {
        ITERATION_ORDER_STRIDE_CASE_RECTILINEAR(RegularMesh3D, 0,1,2)
        ITERATION_ORDER_STRIDE_CASE_RECTILINEAR(RegularMesh3D, 0,2,1)
        ITERATION_ORDER_STRIDE_CASE_RECTILINEAR(RegularMesh3D, 1,0,2)
        ITERATION_ORDER_STRIDE_CASE_RECTILINEAR(RegularMesh3D, 1,2,0)
        ITERATION_ORDER_STRIDE_CASE_RECTILINEAR(RegularMesh3D, 2,0,1)
        ITERATION_ORDER_STRIDE_CASE_RECTILINEAR(RegularMesh3D, 2,1,0)
    }
    return strides;
}


template <typename T, typename MeshT, int dim>
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

    py::incref(oself.ptr()); PyArray_BASE(arr) = oself.ptr(); // Make sure the data vector stays alive as long as the array
    return py::object(py::handle<>(arr));
}


template <typename T, int dim>
static py::object DataVectorWrap__array__(py::object oself) {
    const DataVectorWrap<T,dim>* self = py::extract<const DataVectorWrap<T,dim>*>(oself);

    if (self->mesh_changed) throw Exception("Cannot create array, mesh changed since data retrieval");

    npy_intp dims[] = { self->mesh->size() * get_dim<T>() };

    PyObject* arr = PyArray_SimpleNewFromData(1, dims, detail::get_typenum<T>(), (void*)self->data());
    if (arr == nullptr) throw plask::CriticalException("Cannot create array from data");

    py::incref(oself.ptr()); PyArray_BASE(arr) = oself.ptr(); // Make sure the data vector stays alive as long as the array
    return py::object(py::handle<>(arr));
}




template <typename T, typename MeshT>
static size_t checkMeshAndArray(PyArrayObject* arr, const MeshT& mesh) {
    auto mesh_dims = get_meshdims(mesh);
    if (get_dim<T>() != 1)  mesh_dims.push_back(get_dim<T>());

    if ((size_t)PyArray_NDIM(arr) != mesh_dims.size()) throw ValueError("Provided array must have either 1 or %1% dimensions", MeshT::DIM);

    for (size_t i = 0; i != mesh_dims.size(); ++i)
        if (mesh_dims[i] != PyArray_DIMS(arr)[i])
            throw ValueError("Dimension %1% for the array (%3%) does not match with the mesh (%2%)", i, mesh_dims[i], PyArray_DIMS(arr)[i]);

    auto mesh_strides = get_meshstrides<T>(mesh, mesh_dims.size());
    for (size_t i = 0; i != mesh_dims.size(); ++i)
        if (mesh_strides[i] != PyArray_STRIDES(arr)[i])
            throw ValueError("Stride %1% for the array do not correspond to the current mesh ordering", i);

    return mesh.size();
}

template <typename T>
struct NumpyDataDestructor: public DataVector<T>::Destructor {
    PyArrayObject* arr;
    NumpyDataDestructor(PyArrayObject* arr) : arr(arr) { Py_XINCREF(arr); }
    virtual ~NumpyDataDestructor() { Py_XDECREF(arr); }
    virtual void destruct(T* data) {}
};

template <typename T, int dim>
static py::object makeDataVector(PyArrayObject* arr, shared_ptr<MeshD<dim>> mesh) {

    size_t size;

    auto regular = dynamic_pointer_cast<RectangularMesh<dim,RegularMesh1D>>(mesh);
    auto rectilinear = dynamic_pointer_cast<RectangularMesh<dim,RectilinearMesh1D>>(mesh);

    if (PyArray_NDIM(arr) != 1) {

        if (regular) size = checkMeshAndArray<T, RectangularMesh<dim,RegularMesh1D>>(arr, *regular);
        else if (rectilinear) size = checkMeshAndArray<T, RectangularMesh<dim,RectilinearMesh1D>>(arr, *rectilinear);
        else throw TypeError("For this mesh type only one-dimensional array is allowed");

    } else
        size = PyArray_DIMS(arr)[0] / get_dim<T>();

    if (size != mesh->size()) throw ValueError("Sizes of data (%1%) and mesh (%2%) do not match", size, mesh->size());

    auto data = make_shared<DataVectorWrap<T,dim>>(DataVector<T>((T*)PyArray_DATA(arr), size), mesh);
    data->setDataDestructor(new NumpyDataDestructor<T>(arr));

    return py::object(data);
}

py::object Data(PyObject* obj, py::object omesh) {
    if (!PyArray_Check(obj)) throw TypeError("data needs to be array object");

    PyArrayObject* arr = (PyArrayObject*)obj;

    size_t ndim = PyArray_NDIM(arr);
    size_t last_dim = PyArray_DIMS(arr)[ndim-1];

    try {
        shared_ptr<MeshD<2>> mesh = py::extract<shared_ptr<MeshD<2>>>(omesh);

        switch (PyArray_TYPE(arr)) {
            case NPY_DOUBLE:
                if (ndim == 3) {
                    if (last_dim == 2) return makeDataVector<Vec<2,double>, 2>(arr, mesh);
                    else if (last_dim == 3) return makeDataVector<Vec<2,double>, 2>(arr, mesh);
                } else if (ndim == 1) {
                    if (last_dim == 2 * mesh->size()) return makeDataVector<Vec<2,double>, 2>(arr, mesh);
                    else if (last_dim == 3 * mesh->size()) return makeDataVector<Vec<3,double>, 2>(arr, mesh);
                }
                return makeDataVector<double, 2>(arr, mesh);
            case NPY_CDOUBLE:
                if (ndim == 3) {
                    if (last_dim == 2) return makeDataVector<Vec<2,dcomplex>, 2>(arr, mesh);
                    else if (last_dim == 3) return makeDataVector<Vec<2,dcomplex>, 2>(arr, mesh);
                } else if (ndim == 1) {
                    if (last_dim == 2 * mesh->size()) return makeDataVector<Vec<2,dcomplex>, 2>(arr, mesh);
                    else if (last_dim == 3 * mesh->size()) return makeDataVector<Vec<3,dcomplex>, 2>(arr, mesh);
                }
                return makeDataVector<dcomplex, 2>(arr, mesh);
            default:
                throw TypeError("array has wrong dtype (only float and complex allowed)");
        }

    } catch (py::error_already_set) { PyErr_Clear(); try {
        shared_ptr<MeshD<3>> mesh = py::extract<shared_ptr<MeshD<3>>>(omesh);

        switch (PyArray_TYPE(arr)) {
            case NPY_DOUBLE:
                if (ndim == 4) {
                    if (last_dim == 2) return makeDataVector<Vec<2,double>, 3>(arr, mesh);
                    else if (last_dim == 3) return makeDataVector<Vec<2,double>, 3>(arr, mesh);
                } else if (ndim == 1) {
                    if (last_dim == 2 * mesh->size()) return makeDataVector<Vec<2,double>, 3>(arr, mesh);
                    else if (last_dim == 3 * mesh->size()) return makeDataVector<Vec<3,double>, 3>(arr, mesh);
                }
                return makeDataVector<double, 3>(arr, mesh);
            case NPY_CDOUBLE:
                if (ndim == 4) {
                    if (last_dim == 2) return makeDataVector<Vec<2,dcomplex>, 3>(arr, mesh);
                    else if (last_dim == 3) return makeDataVector<Vec<2,dcomplex>, 3>(arr, mesh);
                } else if (ndim == 1) {
                    if (last_dim == 2 * mesh->size()) return makeDataVector<Vec<2,dcomplex>, 3>(arr, mesh);
                    else if (last_dim == 3 * mesh->size()) return makeDataVector<Vec<3,dcomplex>, 3>(arr, mesh);
                }
                return makeDataVector<dcomplex, 3>(arr, mesh);
            default:
                throw TypeError("array has wrong dtype (only float and complex allowed)");
        }

    } catch (py::error_already_set) {
        throw TypeError("mesh must be a proper mesh object");
    }}

    return py::object();
}


template <typename T, int dim>
void register_data_vector() {

    py::class_<DataVectorWrap<T,dim>, shared_ptr<DataVectorWrap<T,dim>>>("Data", "Data returned by field providers", py::no_init)
        .def_readonly("mesh", &DataVectorWrap<T,dim>::mesh)
        .def("__len__", &DataVectorWrap<T,dim>::size)
        .def("__getitem__", &DataVectorWrap_getitem<T,dim>)
        .def("__getslice__", &DataVectorWrap_getslice<T,dim>)
        .def("__contains__", &DataVectorWrap_contains<T,dim>)
        .def("__iter__", py::range(&DataVectorWrap_begin<T,dim>, &DataVectorWrap_end<T,dim>))
        .def("__array__", &DataVectorWrap__array__<T,dim>)
        .add_static_property("dtype", &DataVector_dtype<T,dim>, "Type of the held values")
        .add_property("array", &DataVectorWrap_Array<T,dim>, "Array formatted by the mesh")
    ;

    py::def("Data", &Data, (py::arg("array"), "mesh"), "Create new data from array and mesh");
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

