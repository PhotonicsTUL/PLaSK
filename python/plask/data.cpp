#include "python_globals.h"
#include "python_provider.h"
#include "python_numpy.h"
#include "python_mesh.h"

#include <plask/mesh/mesh.h>
#include <plask/mesh/rectangular.h>
#include <plask/data.h>

namespace boost { namespace python {

template <>
struct base_type_traits<PyArrayObject>
{
    typedef PyObject type;
};

}}

namespace plask { namespace python {

/*
 * Some helper functions for getting information on rectangular meshes
 */
namespace detail {

    template <typename T> struct basetype { typedef T type; };
    template <typename T, int dim> struct basetype<const Vec<dim,T>> { typedef T type; };
    template <typename T> struct basetype<const Tensor2<T>> { typedef T type; };
    template <typename T> struct basetype<const Tensor3<T>> { typedef T type; };

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
    template <> constexpr inline npy_intp type_dim<const Tensor2<dcomplex>>() { return 2; }
    template <> constexpr inline npy_intp type_dim<const Tensor3<double>>() { return 4; }
    template <> constexpr inline npy_intp type_dim<const Tensor3<dcomplex>>() { return 4; }


    inline static std::vector<npy_intp> mesh_dims(const RectangularMesh<2>& mesh) { return { npy_intp(mesh.axis0->size()), npy_intp(mesh.axis1->size()) }; }
    inline static std::vector<npy_intp> mesh_dims(const RectangularMesh<3>& mesh) { return { npy_intp(mesh.axis0->size()), npy_intp(mesh.axis1->size()), npy_intp(mesh.axis2->size()) }; }

    template <typename T>
    inline static std::vector<npy_intp> mesh_strides(const RectangularMesh<2>& mesh, size_t nd) {
        std::vector<npy_intp> strides(nd);
        strides.back() = sizeof(T) / type_dim<T>();
        if (mesh.getIterationOrder() == RectangularMesh<2>::ORDER_10) {
            strides[0] = sizeof(T);
            strides[1] = mesh.axis0->size() * sizeof(T);
        } else {
            strides[0] = mesh.axis1->size() * sizeof(T);
            strides[1] = sizeof(T);
        }
        return strides;
    }

    #define ITERATION_ORDER_STRIDE_CASE_RECTILINEAR(MeshT, first, second, third) \
        case MeshT::ORDER_##first##second##third: \
            strides[first] = mesh.axis##second->size() * mesh.axis##third->size() * sizeof(T); \
            strides[second] = mesh.axis##third->size() * sizeof(T); \
            strides[third] = sizeof(T); \
            break;

    template <typename T>
    inline static std::vector<npy_intp> mesh_strides(const RectangularMesh<3>& mesh, size_t nd) {
        std::vector<npy_intp> strides(nd, sizeof(T)/type_dim<T>());
        typedef RectangularMesh<3> Mesh3D;
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
    py::object arr(py::handle<>(PyArray_SimpleNew((dims[1]!=1)? 2 : 1, dims, detail::typenum<T>())));
    typename std::remove_const<T>::type* arr_data = static_cast<typename std::remove_const<T>::type*>(PyArray_DATA((PyArrayObject*)arr.ptr()));
    for (auto i = self.begin()+from; i < self.begin()+to; ++i, ++arr_data)
        *arr_data = *i;
    return arr;
}


template <typename T, int dim>
static bool DataVectorWrap_contains(const DataVectorWrap<T,dim>& self, const T& key) { return std::find(self.begin(), self.end(), key) != self.end(); }


template <typename T, int dim>
py::handle<> DataVector_dtype() {
    return detail::dtype<typename std::remove_const<T>::type>();
}



template <typename T, int dim>
static py::object DataVectorWrap__array__(py::object oself, py::object dtype=py::object()) {

    const DataVectorWrap<T,dim>* self = py::extract<const DataVectorWrap<T,dim>*>(oself);

    if (self->mesh_changed) throw Exception("Cannot create array, mesh changed since data retrieval");

    const int nd = (detail::type_dim<T>() == 1)? 1 : 2;

    npy_intp dims[] = { self->mesh->size(), detail::type_dim<T>() };
    npy_intp strides[] = { sizeof(T), sizeof(T) / detail::type_dim<T>() };

    PyObject* arr = PyArray_New(&PyArray_Type, nd, dims, detail::typenum<T>(), strides, (void*)self->data(), 0, 0, NULL);
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

    PyObject* arr = DataVectorWrap_ArrayImpl<T, RectangularMesh<2>>(self);
    if (!arr) arr = DataVectorWrap_ArrayImpl<T, RectangularMesh<3>>(self);

    if (arr == nullptr) throw TypeError("Cannot create array for data on this mesh type (possible only for {0})",
                                        (dim == 2)? "mesh.RectangularMesh2D" : "mesh.RectangularMesh3D");

    py::incref(oself.ptr());
    PyArray_SetBaseObject((PyArrayObject*)arr, oself.ptr()); // Make sure the data vector stays alive as long as the array
    // confirm_array<T>(arr, oself, dtype);

    return py::object(py::handle<>(arr));
}

template <typename T, int dim> DataVectorWrap<T,dim>
dataInterpolate(const DataVectorWrap<T,dim>& self, shared_ptr<MeshD<dim>> dst_mesh, InterpolationMethod method);


namespace detail {

    struct NumpyDataDeleter {
        PyArrayObject* arr;
        NumpyDataDeleter(PyArrayObject* arr) : arr(arr) {
            OmpLockGuard<OmpNestLock> lock(python_omp_lock);
            Py_XINCREF(arr);
        }
        void operator()(void*) {
            OmpLockGuard<OmpNestLock> lock(python_omp_lock);
            Py_XDECREF(arr);
        }
    };

    template <typename T, int dim>
    static py::object makeDataVectorImpl(PyArrayObject* arr, shared_ptr<MeshD<dim>> mesh) {

        size_t size;
        py::handle<PyArrayObject> newarr;

        if (PyArray_NDIM(arr) != 1) {
            auto rectangular = dynamic_pointer_cast<RectangularMesh<dim>>(mesh);
            if (!rectangular) throw TypeError("For this mesh type only one-dimensional array is allowed");

            auto meshdims = mesh_dims(*rectangular);
            if (type_dim<T>() != 1)  meshdims.push_back(type_dim<T>());
            size_t nd = meshdims.size();

            if ((size_t)PyArray_NDIM(arr) != nd) throw ValueError("Provided array must have either 1 or {0} dimensions", dim);

            for (size_t i = 0; i != nd; ++i)
                if (meshdims[i] != PyArray_DIMS(arr)[i])
                    throw ValueError("Dimension {0} for the array ({2}) does not match with the mesh ({1})", i, meshdims[i], PyArray_DIMS(arr)[i]);

            auto meshstrides = mesh_strides<T>(*rectangular, nd);
            for (size_t i = 0; i != nd; ++i) {
                if (meshstrides[i] != PyArray_STRIDES(arr)[i]) {
                    writelog(LOG_DEBUG, "Copying numpy array to match mesh strides");

                    newarr = py::handle<PyArrayObject>(
                        (PyArrayObject*)PyArray_New(&PyArray_Type, nd, meshdims.data(),
                                                    PyArray_TYPE(arr), meshstrides.data(),
                                                    nullptr, 0, 0, nullptr)
                    );

                    PyArray_CopyInto(newarr.get(), arr);
                    arr = newarr.get();
                    break;
                }
            }

            size = mesh->size();

        } else
            size = PyArray_DIMS(arr)[0] / type_dim<T>();

        if (size != mesh->size()) throw ValueError("Sizes of data ({0}) and mesh ({1}) do not match", size, mesh->size());

        auto data = plask::make_shared<DataVectorWrap<const T,dim>>(
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
            else if (last_dim == 3) return makeDataVectorImpl<Vec<3,T>, dim>(arr, mesh);
            else if (last_dim == 4) return makeDataVectorImpl<Tensor3<T>, dim>(arr, mesh);
        } else if (ndim == 1) {
            if (last_dim == 2 * mesh->size()) return makeDataVectorImpl<Vec<2,T>, dim>(arr, mesh);
            else if (last_dim == 3 * mesh->size()) return makeDataVectorImpl<Vec<3,T>, dim>(arr, mesh);
            else if (last_dim == 4 * mesh->size()) return makeDataVectorImpl<Tensor3<T>, dim>(arr, mesh);
        }
        return makeDataVectorImpl<T, dim>(arr, mesh);
    }

} // namespace  detail

PLASK_PYTHON_API py::object Data(PyObject* obj, py::object omesh) {
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
static DataVectorWrap<T,dim> DataVectorWrap__add__(const DataVectorWrap<T,dim>& vec1, const DataVectorWrap<T,dim>& vec2) {
    if (vec1.mesh != vec2.mesh)
        throw ValueError("You may only add data on the same mesh");
    return DataVectorWrap<T,dim>(vec1 + vec2, vec1.mesh);
}

template <typename T, int dim>
static DataVectorWrap<T,dim> DataVectorWrap__sub__(const DataVectorWrap<T,dim>& vec1, const DataVectorWrap<T,dim>& vec2) {
    if (vec1.mesh != vec2.mesh)
        throw ValueError("You may only subtract data on the same mesh");
    return DataVectorWrap<T,dim>(vec1 + vec2, vec1.mesh);
}

// template <typename T, int dim>
// static void DataVectorWrap__iadd__(const DataVectorWrap<T,dim>& vec1, const DataVectorWrap<T,dim>& vec2) {
//     if (vec1.mesh != vec2.mesh)
//         throw ValueError("You may only add data on the same mesh");
//     vec1 += vec2;
// }
//
// template <typename T, int dim>
// static void DataVectorWrap__isub__(const DataVectorWrap<T,dim>& vec1, const DataVectorWrap<T,dim>& vec2) {
//     if (vec1.mesh != vec2.mesh)
//         throw ValueError("You may only subtract data on the same mesh");
//     vec1 -= vec2;
// }

template <typename T, int dim>
static DataVectorWrap<T,dim> DataVectorWrap__neg__(const DataVectorWrap<T,dim>& vec) {
    return DataVectorWrap<T,dim>(-vec, vec.mesh);
}

template <typename T, int dim>
static DataVectorWrap<T,dim> DataVectorWrap__mul__(const DataVectorWrap<T,dim>& vec, typename detail::basetype<T>::type a) {
    return DataVectorWrap<T,dim>(vec * a, vec.mesh);
}

template <typename T, int dim>
static DataVectorWrap<T,dim> DataVectorWrap__div__(const DataVectorWrap<T,dim>& vec, typename detail::basetype<T>::type a) {
    return DataVectorWrap<T,dim>(vec / a, vec.mesh);
}

// template <typename T, int dim>
// static void DataVectorWrap__imul__(const DataVectorWrap<T,dim>& vec, typename detail::basetype<T>::type a) {
//     vec *= a;
// }
//
// template <typename T, int dim>
// void DataVectorWrap__idiv__(const DataVectorWrap<T,dim>& vec, typename detail::basetype<T>::type a) {
//     vec /= a;
// }

// template <typename T, int dim>
// static DataVectorWrap<T,dim> DataVectorWrap__abs__(const DataVectorWrap<T,dim>& vec) {
//     return DataVectorWrap<T,dim>(abs(vec), vec.mesh);
// }

template <typename T, int dim>
static bool DataVectorWrap__eq__(const DataVectorWrap<T,dim>& vec1, const DataVectorWrap<T,dim>& vec2) {
    if (vec1.mesh != vec2.mesh) return false;
    for (size_t i = 0; i < vec1.size(); ++i)
        if (vec1[i] != vec2[i]) return false;
    return true;
}

template <typename T, int dim>
void register_data_vector() {

    py::class_<DataVectorWrap<const T,dim>, shared_ptr<DataVectorWrap<const T,dim>>>
    data("_Data",
    "Data returned by field providers.\n\n"
    "This class is returned by field providers and receivers and cointains the values\n"
    "of the computed field at specified mesh points. It can be passed to the field\n"
    "plotting and saving functions or even feeded to some receivers. Also, if the\n"
    "mesh is a rectangular one, the data can be converted into an multi-dimensional\n"\
    "numpy array.\n\n"

    "You may access the data by indexing the :class:`~plask.Data` object, where the\n"
    "index always corresponds to the index of the mesh point where the particular\n"
    "value is specified. Hence, you may also iterate :class:`~plask.Data` objects as\n"
    "normal Python sequences.\n\n"

    "You may construct the data object manually from a numpy array and a mesh.\n"
    "The constructor always take two argumentsa as specified below:\n\n"

    "Args:\n"
    "    array: The array with a custom data.\n"
    "        It must be either a one dimensional array with sequential data of the\n"
    "        desired type corresponding to the sequential mesh points or (for the\n"
    "        rectangular meshes) an array with the same shape as returned by the\n"
    "        :attr:`array` attribute.\n"
    "    mesh: The mesh specifying where the data points are located.\n"
    "        The size of the mesh must be equal to the size of the provided array.\n"
    "        Furthermore, when constructing the data from the structured array, the\n"
    "        mesh ordering must match the data stride, so it is possible to avoid\n"
    "        data copying (defaults for both are fine).\n"

    "Examples:\n"
    "    To create the data from the flat sequential array:\n\n"

    "    >>> msh = plask.mesh.Rectangular2D(plask.mesh.Rectilinear([1, 2, 3]),\n"
    "    ... plask.mesh.Rectilinear([10, 20]))\n"
    "    >>> Data(array([1., 2., 3., 4., 5., 6.]), msh)\n"
    "    <plask.Data at 0x4698938>\n\n"

    "    As the ``msh`` is a rectangular mesh, the data can be created from the\n"
    "    structured array with the shape (3, 2), as the first and second mesh\n"
    "    dimensions are 3 and 2, respectively:\n\n"

    "    >>> dat = Data(array([[1., 2.], [3., 4.], [5., 6.]]), msh)\n"
    "    >>> dat[0]\n"
    "    1.0\n\n"

    "    By adding one more dimension, you can create an array of vectors:\n\n"

    "    >>> d = Data(array([[[1.,0.], [2.,0.]], [[3.,0.], [4.,1.]],\n"
    "    ...                 [[5.,1.], [6.,1.]]]), msh)\n"
    "    >>> d.dtype\n"
    "    plask.vec\n"
    "    >>> d[1]\n"
    "    plask.vec(2, 0)\n"
    "    >>> d.array[:,:,0]    # retrieve first components of all the vectors\n"
    "    array([[1., 2.], [3., 4.], [5., 6.]])\n\n"

    "Construction of the data objects is efficient i.e. no data is copied in the\n"
    "memory from the provided array.\n\n",
    py::no_init);
    data
        .def_readonly("mesh", &DataVectorWrap<const T,dim>::mesh,
            "The mesh at which the data was obtained.\n\n"

            "The sequential points of this mesh always correspond to the sequential points of\n"
            "the data. This implies that ``len(data.mesh) == len(data)`` is always True.\n"
         )
        .def("__len__", &DataVectorWrap<const T,dim>::size)
        .def("__getitem__", &DataVectorWrap_getitem<const T,dim>)
        .def("__getslice__", &DataVectorWrap_getslice<const T,dim>)
        .def("__contains__", &DataVectorWrap_contains<const T,dim>)
        .def("__iter__", py::range(&DataVectorWrap_begin<const T,dim>, &DataVectorWrap_end<const T,dim>))
        .def("__array__", &DataVectorWrap__array__<const T,dim>, py::arg("dtype")=py::object())
        .add_property("array", &DataVectorWrap_Array<const T,dim>,
            "Array formatted by the mesh.\n\n"

            "This attribute is available only if the :attr:`mesh` is a rectangular one. It\n"
            "contains the held data reshaped to match the shape of the mesh (i.e. the first\n"
            "dimension is equal the size of the first mesh axis and so on). If the data type\n"
            "is :class:`plask.vec` then the array has one additional dimention equal to 2 for\n"
            "2D vectors and 3 for 3D vectors. The vector components are stored in this\n"
            "dimention.\n\n"

            "Example:\n"
            "    >>> msh = plask.mesh.Rectangular2D(plask.mesh.Rectilinear([1, 2]),\n"
            "    ... plask.mesh.Rectilinear([10, 20]))\n"
            "    >>> dat = Data(array([[[1., 0.], [2., 0.]], [[3., 1.], [4., 1.]]]), msh)\n"
            "    >>> dat.array[:,:,0]\n"
            "    array([[1., 2.],\n"
            "           [3., 4.]])\n\n"

            "Accessing this field is efficient, as only the numpy array view is created and\n"
            "no data is copied in the memory.\n"
         )
        .def("interpolate", dataInterpolate<const T,dim>, (py::arg("mesh"), "interpolation"),
            "Interpolate data to a different mesh.\n\n"

            "This method interpolated data into a different mesh using specified\n"
            "interpolation method. This is exactly the same interpolation that is\n"
            "usually done by solvers in their providers.\n\n"

            "Args:\n"
            "    mesh: Mesh to interpolate into.\n"
            "    interpolation: Requested interpolation method.\n"
            "Returns:\n"
            "    plask._Data: Interpolated data."
        )
        .add_static_property("dtype", &DataVector_dtype<const T,dim>, "Type of the held values.")
        .def("__add__", &DataVectorWrap__add__<const T,dim>)
        .def("__sub__", &DataVectorWrap__sub__<const T,dim>)
        .def("__mul__", &DataVectorWrap__mul__<const T,dim>)
        .def("__rmul__", &DataVectorWrap__mul__<const T,dim>)
        .def("__div__", &DataVectorWrap__div__<const T,dim>)
        .def("__truediv__", &DataVectorWrap__div__<const T,dim>)
        .def("__neg__", &DataVectorWrap__neg__<const T,dim>)
        // .def("__abs__", &DataVectorWrap__abs__<const T,dim>)
        // .def("__iadd__", &DataVectorWrap__iadd__<const T,dim>)
        // .def("__isub__", &DataVectorWrap__isub__<const T,dim>)
        // .def("__imul__", &DataVectorWrap__imul__<const T,dim>)
        // .def("__idiv__", &DataVectorWrap__idiv__<const T,dim>)
        // .def("__itruediv__", &DataVectorWrap__idiv__<const T,dim>)
        .def("__eq__", &DataVectorWrap__eq__<const T,dim>)
    ;
    data.attr("__name__") = "Data";
    data.attr("__module__") = "plask";
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
    register_data_vector<Tensor2<dcomplex>, dim>();
    register_data_vector<Tensor3<double>, dim>();
    register_data_vector<Tensor3<dcomplex>, dim>();
}

void register_data_vectors() {
    // Initialize numpy
    if (!plask_import_array()) throw(py::error_already_set());

    register_data_vectors_d<2>();
    register_data_vectors_d<3>();

    py::def("Data", &Data, (py::arg("array"), "mesh"),

            "Data returned by field providers.\n\n"

            "This class is returned by field providers and receivers and cointains the values\n"
            "of the computed field at specified mesh points. It can be passed to the field\n"
            "plotting and saving functions or even feeded to some receivers. Also, if the\n"
            "mesh is a rectangular one, the data can be converted into an multi-dimensional\n"\
            "numpy array.\n\n"

            "You may access the data by indexing the :class:`~plask.Data` object, where the\n"
            "index always corresponds to the index of the mesh point where the particular\n"
            "value is specified. Hence, you may also iterate :class:`~plask.Data` objects as\n"
            "normal Python sequences.\n\n"

            "You may construct the data object manually from a numpy array and a mesh.\n"
            "The constructor always take two argumentsa as specified below:\n\n"

            "Args:\n"
            "    array: The array with a custom data.\n"
            "        It must be either a one dimensional array with sequential data of the\n"
            "        desired type corresponding to the sequential mesh points or (for the\n"
            "        rectangular meshes) an array with the same shape as returned by the\n"
            "        :attr:`array` attribute.\n"
            "    mesh: The mesh specifying where the data points are located.\n"
            "        The size of the mesh must be equal to the size of the provided array.\n"
            "        Furthermore, when constructing the data from the structured array, the\n"
            "        mesh ordering must match the data stride, so it is possible to avoid\n"
            "        data copying (defaults for both are fine).\n"
            "Returns:\n"
            "    plask._Data: Data based on the specified mesh and array.\n\n"

            "Examples:\n"
            "    To create the data from the flat sequential array:\n\n"

            "    >>> msh = plask.mesh.Rectangular2D(plask.mesh.Rectilinear([1, 2, 3]),\n"
            "    ... plask.mesh.Rectilinear([10, 20]))\n"
            "    >>> Data(array([1., 2., 3., 4., 5., 6.]), msh)\n"
            "    <plask.Data at 0x4698938>\n\n"

            "    As the ``msh`` is a rectangular mesh, the data can be created from the\n"
            "    structured array with the shape (3, 2), as the first and second mesh\n"
            "    dimensions are 3 and 2, respectively:\n\n"

            "    >>> dat = Data(array([[1., 2.], [3., 4.], [5., 6.]]), msh)\n"
            "    >>> dat[0]\n"
            "    1.0\n\n"

            "    By adding one more dimension, you can create an array of vectors:\n\n"

            "    >>> d = Data(array([[[1.,0.], [2.,0.]], [[3.,0.], [4.,1.]],\n"
            "    ...                 [[5.,1.], [6.,1.]]]), msh)\n"
            "    >>> d.dtype\n"
            "    plask.vec\n"
            "    >>> d[1]\n"
            "    plask.vec(2, 0)\n"
            "    >>> d.array[:,:,0]    # retrieve first components of all the vectors\n"
            "    array([[1., 2.], [3., 4.], [5., 6.]])\n\n"

            "Construction of the data objects is efficient i.e. no data is copied in the\n"
            "memory from the provided array.\n\n"
           );
}


} // namespace python

template <int dim, typename SrcT, typename DstT>
struct __InterpolateMeta__<python::MeshWrap<dim>, SrcT, DstT, 0>
{
    inline static LazyData<typename std::remove_const<DstT>::type> interpolate(
            const shared_ptr<const python::MeshWrap<dim>>& src_mesh, const DataVector<const SrcT>& src_vec,
            const shared_ptr<const MeshD<dim>>& dst_mesh, InterpolationMethod method, const InterpolationFlags& flags) {
        OmpLockGuard<OmpNestLock> lock(python::python_omp_lock);
        typedef python::DataVectorWrap<const DstT, dim> ReturnedType;
        boost::python::object omesh(const_pointer_cast<MeshD<dim>>(dst_mesh));
        auto source = plask::make_shared<python::DataVectorWrap<const SrcT, dim>>(
            src_vec, const_pointer_cast<python::MeshWrap<dim>>(src_mesh));
        boost::python::object result = 
            src_mesh->template call_python<boost::python::object>("interpolate", source, omesh, method);
        try {
            return boost::python::extract<ReturnedType>(result)();
        } catch (boost::python::error_already_set) {
            PyErr_Clear();
            return boost::python::extract<ReturnedType>(python::Data(result.ptr(), omesh))();
        }
    }
};

namespace python {

template <typename T, int dim>
PLASK_PYTHON_API DataVectorWrap<T,dim> dataInterpolate(
    const DataVectorWrap<T,dim>& self, shared_ptr<MeshD<dim>> dst_mesh, InterpolationMethod method) {

    if (self.mesh_changed) throw Exception("Cannot interpolate, mesh changed since data retrieval");

    if (auto src_mesh = dynamic_pointer_cast<RectangularMesh<dim>>(self.mesh))
        return DataVectorWrap<T,dim>(interpolate(src_mesh, self, dst_mesh, method), dst_mesh);
    else if (auto src_mesh = dynamic_pointer_cast<MeshWrap<dim>>(self.mesh))
        return DataVectorWrap<T,dim>(interpolate(src_mesh, self, dst_mesh, method), dst_mesh);
    // TODO add new mesh types here

    throw NotImplemented(format("interpolate(source mesh type: {}, interpolation method: {})",
                                typeid(*self.mesh).name(), interpolationMethodNames[method]));
}

}} // namespace plask::python

#include <plask/data.h>
