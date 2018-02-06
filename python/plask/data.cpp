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

namespace plask {

template <> template <>
DataVector<const Tensor2<double>>::DataVector(const DataVector<const Vec<2,double>>& src)
    : DataVector(reinterpret_cast<const DataVector<const Tensor2<double>>&>(src)) {
}

template <> template <>
DataVector<const Tensor2<dcomplex>>::DataVector(const DataVector<const Vec<2,dcomplex>>& src)
    : DataVector(reinterpret_cast<const DataVector<const Tensor2<dcomplex>>&>(src)) {
}


template <typename T>
inline static Tensor2<double> abs(const Tensor2<T>& x) {
    return Tensor2<double>(abs(x.c00), abs(x.c11));
}

template <typename T>
inline static Tensor3<double> abs(const Tensor3<T>& x) {
    return Tensor3<double>(abs(x.c00), abs(x.c11), abs(x.c22), abs(x.c01));
}


template <typename T>
inline static Vec<2,double> real(const Vec<2,T>& x) {
    return Vec<2,double>(real(x.c0), real(x.c1));
}

template <typename T>
inline static Vec<3,double> real(const Vec<3,T>& x) {
    return Vec<3,double>(real(x.c0), real(x.c1), real(x.c2));
}

template <typename T>
inline static Tensor2<double> real(const Tensor2<T>& x) {
    return Tensor2<double>(real(x.c00), real(x.c11));
}

template <typename T>
inline static Tensor3<double> real(const Tensor3<T>& x) {
    return Tensor3<double>(real(x.c00), real(x.c11), real(x.c22), real(x.c01));
}

template <typename T>
inline static Vec<2,double> imag(const Vec<2,T>& x) {
    return Vec<2,double>(imag(x.c0), imag(x.c1));
}

template <typename T>
inline static Vec<3,double> imag(const Vec<3,T>& x) {
    return Vec<3,double>(imag(x.c0), imag(x.c1), imag(x.c2));
}

template <typename T>
inline static Tensor2<double> imag(const Tensor2<T>& x) {
    return Tensor2<double>(imag(x.c00), imag(x.c11));
}

template <typename T>
inline static Tensor3<double> imag(const Tensor3<T>& x) {
    return Tensor3<double>(imag(x.c00), imag(x.c11), imag(x.c22), imag(x.c01));
}


namespace python {

static const char* DATA_DOCSTRING =
    u8"Data returned by field providers.\n\n"

    u8"This class is returned by field providers and receivers and cointains the values\n"
    u8"of the computed field at specified mesh points. It can be passed to the field\n"
    u8"plotting and saving functions or even feeded to some receivers. Also, if the\n"
    u8"mesh is a rectangular one, the data can be converted into an multi-dimensional\n"\
    u8"numpy array.\n\n"

    u8"You may access the data by indexing the :class:`~plask.Data` object, where the\n"
    u8"index always corresponds to the index of the mesh point where the particular\n"
    u8"value is specified. Hence, you may also iterate :class:`~plask.Data` objects as\n"
    u8"normal Python sequences.\n\n"

    u8"You may construct the data object manually from a numpy array and a mesh.\n"
    u8"The constructor always take two argumentsa as specified below:\n\n"

    u8"Args:\n"
    u8"    array: The array with a custom data.\n"
    u8"        It must be either a one dimensional array with sequential data of the\n"
    u8"        desired type corresponding to the sequential mesh points or (for the\n"
    u8"        rectangular meshes) an array with the same shape as returned by the\n"
    u8"        :attr:`array` attribute.\n"
    u8"    mesh: The mesh specifying where the data points are located.\n"
    u8"        The size of the mesh must be equal to the size of the provided array.\n"
    u8"        Furthermore, when constructing the data from the structured array, the\n"
    u8"        mesh ordering must match the data stride, so it is possible to avoid\n"
    u8"        data copying (defaults for both are fine).\n"
    u8"Returns:\n"
    u8"    plask._Data: Data based on the specified mesh and array.\n\n"

    u8"Examples:\n"
    u8"    To create the data from the flat sequential array:\n\n"

    u8"    >>> msh = plask.mesh.Rectangular2D(plask.mesh.Rectilinear([1, 2, 3]),\n"
    u8"    ... plask.mesh.Rectilinear([10, 20]))\n"
    u8"    >>> Data(array([1., 2., 3., 4., 5., 6.]), msh)\n"
    u8"    <plask.Data at 0x4698938>\n\n"

    u8"    As the ``msh`` is a rectangular mesh, the data can be created from the\n"
    u8"    structured array with the shape (3, 2), as the first and second mesh\n"
    u8"    dimensions are 3 and 2, respectively:\n\n"

    u8"    >>> dat = Data(array([[1., 2.], [3., 4.], [5., 6.]]), msh)\n"
    u8"    >>> dat[0]\n"
    u8"    1.0\n\n"

    u8"    By adding one more dimension, you can create an array of vectors:\n\n"

    u8"    >>> d = Data(array([[[1.,0.], [2.,0.]], [[3.,0.], [4.,1.]],\n"
    u8"    ...                 [[5.,1.], [6.,1.]]]), msh)\n"
    u8"    >>> d.dtype\n"
    u8"    plask.vec\n"
    u8"    >>> d[1]\n"
    u8"    plask.vec(2, 0)\n"
    u8"    >>> d.array[:,:,0]    # retrieve first components of all the vectors\n"
    u8"    array([[1., 2.], [3., 4.], [5., 6.]])\n\n"

    u8"Construction of the data objects is efficient i.e. no data is copied in the\n"
    u8"memory from the provided array.\n";


/*
 * Some helper functions for getting information on rectangular meshes
 */
namespace detail {

    template <typename T> struct isBasicData: std::false_type {};
    template <typename T> struct isBasicData<const T>: isBasicData<typename std::remove_const<T>::type> {};

    template<> struct isBasicData<double> : std::true_type {};
    template<> struct isBasicData<dcomplex> : std::true_type {};
    template<> struct isBasicData<Vec<2,double>> : std::true_type {};
    template<> struct isBasicData<Vec<2,dcomplex>> : std::true_type {};
    template<> struct isBasicData<Vec<3,double>> : std::true_type {};
    template<> struct isBasicData<Vec<3,dcomplex>> : std::true_type {};
    template<> struct isBasicData<Tensor2<double>> : std::true_type {};
    template<> struct isBasicData<Tensor2<dcomplex>> : std::true_type {};
    template<> struct isBasicData<Tensor3<double>> : std::true_type {};
    template<> struct isBasicData<Tensor3<dcomplex>> : std::true_type {};

    template <typename T> struct basetype { typedef T type; };
    template <typename T, int dim> struct basetype<const Vec<dim,T>> { typedef T type; };
    template <typename T> struct basetype<const Tensor2<T>> { typedef T type; };
    template <typename T> struct basetype<const Tensor3<T>> { typedef T type; };

    template <typename T> constexpr inline static npy_intp type_dim() { return 1; }
    template <> constexpr inline npy_intp type_dim<Vec<2,double>>() { return 2; }
    template <> constexpr inline npy_intp type_dim<Vec<2,dcomplex>>() { return 2; }
    template <> constexpr inline npy_intp type_dim<Vec<3,double>>() { return 3; }
    template <> constexpr inline npy_intp type_dim<Vec<3,dcomplex>>() { return 3; }
    // template <> constexpr inline npy_intp type_dim<Tensor2<double>>() { return 2; }
    // template <> constexpr inline npy_intp type_dim<Tensor2<dcomplex>>() { return 2; }
    template <> constexpr inline npy_intp type_dim<Tensor3<double>>() { return 4; }
    template <> constexpr inline npy_intp type_dim<Tensor3<dcomplex>>() { return 4; }
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
static const typename DataVector<T>::const_iterator PythonDataVector_begin(const PythonDataVector<T,dim>& self) { return self.begin(); }

template <typename T, int dim>
static const typename DataVector<T>::const_iterator PythonDataVector_end(const PythonDataVector<T,dim>& self) { return self.end(); }

template <typename T, int dim>
static T PythonDataVector_getitem(const PythonDataVector<T,dim>& self, std::ptrdiff_t i) {
    if (i < 0) i += self.size();
    if (i < 0 || std::size_t(i) >= self.size()) throw IndexError(u8"index out of range");
    return self[i];
}


template <typename T, int dim>
static typename std::enable_if<detail::isBasicData<T>::value, py::object>::type
PythonDataVector_getslice(const PythonDataVector<T,dim>& self, std::ptrdiff_t from, std::ptrdiff_t to) {
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
static typename std::enable_if<!detail::isBasicData<T>::value, py::object>::type
PythonDataVector_getslice(const PythonDataVector<T,dim>& self, std::ptrdiff_t from, std::ptrdiff_t to) {
    if (from < 0) from = self.size() - from;
    if (to < 0) to = self.size() - to;
    if (from < 0) from = 0;
    if (std::size_t(to) > self.size()) to = self.size();

    npy_intp dims[] = { to-from };
    py::object arr(py::handle<>(PyArray_SimpleNew(1, dims, NPY_OBJECT)));
    PyObject** arr_data = static_cast<PyObject**>(PyArray_DATA((PyArrayObject*)arr.ptr()));
    for (auto i = self.begin()+from; i < self.begin()+to; ++i, ++arr_data)
        *arr_data = py::incref(py::object(*i).ptr());
    return arr;
}


template <typename T, int dim>
static bool PythonDataVector_contains(const PythonDataVector<T,dim>& self, const T& key) { return std::find(self.begin(), self.end(), key) != self.end(); }


template <typename T, int dim>
py::handle<> DataVector_dtype() {
    return detail::dtype<typename std::remove_const<T>::type>();
}


template <typename T, int dim>
static typename std::enable_if<detail::isBasicData<T>::value, py::object>::type
PythonDataVector__array__(py::object oself, py::object dtype=py::object()) {

    const PythonDataVector<T,dim>* self = py::extract<const PythonDataVector<T,dim>*>(oself);

    if (self->mesh_changed) throw Exception(u8"Cannot create array, mesh changed since data retrieval");

    const int nd = (detail::type_dim<T>() == 1)? 1 : 2;

    npy_intp dims[] = { static_cast<npy_intp>(self->mesh->size()), detail::type_dim<T>() };
    npy_intp strides[] = { sizeof(T), sizeof(T) / detail::type_dim<T>() };

    PyObject* arr = PyArray_New(&PyArray_Type, nd, dims, detail::typenum<T>(), strides, (void*)self->data(), 0, 0, NULL);
    if (arr == nullptr) throw plask::CriticalException(u8"Cannot create array from data");

    confirm_array<T>(arr, oself, dtype);

    return py::object(py::handle<>(arr));
}

template <typename T, int dim>
static typename std::enable_if<!detail::isBasicData<T>::value, py::object>::type
PythonDataVector__array__(const PythonDataVector<T,dim>& self, py::object dtype=py::object()) {

    if (dtype != py::object()) throw ValueError(u8"dtype for this data must not be specified");

    if (self.mesh_changed) throw Exception(u8"Cannot create array, mesh changed since data retrieval");

    return PythonDataVector_getslice(self, 0, self.size());
}



template <typename T, typename MeshT, int dim>
static inline typename std::enable_if<detail::isBasicData<T>::value, PyObject*>::type
PythonDataVector_ArrayImpl(const PythonDataVector<T,dim>* self) {
    shared_ptr<MeshT> mesh = dynamic_pointer_cast<MeshT>(self->mesh);
    if (!mesh) return nullptr;

    std::vector<npy_intp> dims = detail::mesh_dims(*mesh);
    if (detail::type_dim<T>() != 1) dims.push_back(detail::type_dim<T>());

    PyObject* arr = PyArray_New(&PyArray_Type,
                                int(dims.size()),
                                & dims.front(),
                                detail::typenum<T>(),
                                & detail::mesh_strides<T>(*mesh, dims.size()).front(),
                                (void*)self->data(),
                                0,
                                0,
                                NULL);
    if (arr == nullptr) throw plask::CriticalException(u8"Cannot create array from data");
    return arr;
}

template <typename T, typename MeshT, int dim>
static inline typename std::enable_if<!detail::isBasicData<T>::value, PyObject*>::type
PythonDataVector_ArrayImpl(const PythonDataVector<T,dim>* self) {
    shared_ptr<MeshT> mesh = dynamic_pointer_cast<MeshT>(self->mesh);
    if (!mesh) return nullptr;

    std::vector<npy_intp> dims = detail::mesh_dims(*mesh);

    PyObject* arr = PyArray_SimpleNew(int(dims.size()), &dims.front(), NPY_OBJECT);
    if (arr == nullptr) throw plask::CriticalException(u8"Cannot create array from data");

    PyObject** arr_data = static_cast<PyObject**>(PyArray_DATA((PyArrayObject*)arr));
    for (auto i = self->begin(); i < self->end(); ++i, ++arr_data)
        *arr_data = py::incref(py::object(*i).ptr());

    return arr;
}

template <typename T, int dim>
static py::object PythonDataVector_Array(py::object oself) {
    const PythonDataVector<T,dim>* self = py::extract<const PythonDataVector<T,dim>*>(oself);

    if (self->mesh_changed) throw Exception(u8"Cannot create array, mesh changed since data retrieval");

    PyObject* arr = PythonDataVector_ArrayImpl<T, RectangularMesh<2>>(self);
    if (!arr) arr = PythonDataVector_ArrayImpl<T, RectangularMesh<3>>(self);

    if (arr == nullptr) throw TypeError(u8"Cannot create array for data on this mesh type (possible only for {0})",
                                        (dim == 2)? "mesh.RectangularMesh2D" : "mesh.RectangularMesh3D");

    py::incref(oself.ptr());
    PyArray_SetBaseObject((PyArrayObject*)arr, oself.ptr()); // Make sure the data vector stays alive as long as the array
    // confirm_array<T>(arr, oself, dtype);

    return py::object(py::handle<>(arr));
}


namespace detail {

    template <typename T, int dim>
    static typename std::enable_if<detail::isBasicData<T>::value, py::object>::type
    makeDataVectorImpl(PyArrayObject* arr, shared_ptr<MeshD<dim>> mesh) {

        size_t size;
        py::handle<PyArrayObject> newarr;

        if (PyArray_NDIM(arr) == 1) {
            size = PyArray_DIMS(arr)[0] / type_dim<T>();
            if (PyArray_STRIDES(arr)[0] != sizeof(T)) {
                writelog(LOG_DEBUG, u8"Copying numpy array to make is contiguous");
                npy_intp sizes[] = { PyArray_DIMS(arr)[0] };
                npy_intp strides[] = { sizeof(T) };
                newarr = py::handle<PyArrayObject>(
                    (PyArrayObject*)PyArray_New(&PyArray_Type, 1, sizes,
                                                PyArray_TYPE(arr), strides,
                                                nullptr, 0, 0, nullptr)
                );
                PyArray_CopyInto(newarr.get(), arr);
                arr = newarr.get();
            }
        } else if (type_dim<T>() != 1 && PyArray_NDIM(arr) == 2 &&
                   std::size_t(PyArray_DIMS(arr)[0]) == mesh->size() && PyArray_DIMS(arr)[1] == type_dim<T>()) {
            size = mesh->size();
            if (PyArray_STRIDES(arr)[0] != sizeof(T)) {
                writelog(LOG_DEBUG, u8"Copying numpy array to make is contiguous");
                npy_intp sizes[] = { static_cast<npy_intp>(size), type_dim<T>() };
                npy_intp strides[] = { sizeof(T), sizeof(T) / type_dim<T>() };
                newarr = py::handle<PyArrayObject>(
                    (PyArrayObject*)PyArray_New(&PyArray_Type, 2, sizes,
                                                PyArray_TYPE(arr), strides,
                                                nullptr, 0, 0, nullptr)
                );
                PyArray_CopyInto(newarr.get(), arr);
                arr = newarr.get();
            }
        } else {
            auto rectangular = dynamic_pointer_cast<RectangularMesh<dim>>(mesh);
            if (!rectangular) throw TypeError(u8"For this mesh type only one-dimensional array is allowed");
            auto meshdims = mesh_dims(*rectangular);
            if (type_dim<T>() != 1) meshdims.push_back(type_dim<T>());
            std::size_t nd = meshdims.size();
            if ((std::size_t)PyArray_NDIM(arr) != nd) throw ValueError(u8"Provided array must have either 1 or {0} dimensions", dim);
            for (std::size_t i = 0; i != nd; ++i)
                if (meshdims[i] != PyArray_DIMS(arr)[i])
                    throw ValueError(u8"Dimension {0} for the array ({2}) does not match with the mesh ({1})", i, meshdims[i], PyArray_DIMS(arr)[i]);
            auto meshstrides = mesh_strides<T>(*rectangular, nd);
            for (std::size_t i = 0; i != nd; ++i) {
                if (meshstrides[i] != PyArray_STRIDES(arr)[i]) {
                    writelog(LOG_DEBUG, u8"Copying numpy array to match mesh strides");
                    newarr = py::handle<PyArrayObject>(
                        (PyArrayObject*)PyArray_New(&PyArray_Type, int(nd), meshdims.data(),
                                                    PyArray_TYPE(arr), meshstrides.data(),
                                                    nullptr, 0, 0, nullptr)
                    );
                    PyArray_CopyInto(newarr.get(), arr);
                    arr = newarr.get();
                    break;
                }
            }
            size = mesh->size();
        }

        if (size != mesh->size()) throw ValueError(u8"Sizes of data ({0}) and mesh ({1}) do not match", size, mesh->size());

        auto result = plask::make_shared<PythonDataVector<const T,dim>>(
            DataVector<const T>((const T*)PyArray_DATA(arr), size, NumpyDataDeleter(arr)),
            mesh);

        return py::object(result);
    }

    template <typename T, int dim>
    static typename std::enable_if<!detail::isBasicData<T>::value, py::object>::type
    makeDataVectorImpl(PyArrayObject* arr, shared_ptr<MeshD<dim>> mesh) {

        size_t size;
        py::handle<PyArrayObject> newarr;

        if (PyArray_NDIM(arr) != 1) {
            throw NotImplemented(u8"Data from multi-dimensional array for this dtype");
        } else {
            if (size != mesh->size()) throw ValueError(u8"Sizes of data ({0}) and mesh ({1}) do not match", size, mesh->size());
            size = PyArray_DIMS(arr)[0];
            PyArrayIterObject* iter = (PyArrayIterObject*)PyArray_IterNew((PyObject*)arr);
            auto data = DataVector<T>(size);
            for (size_t i = 0; i != size; ++i) {
                data[i] = py::extract<T>((*(PyObject**)iter->dataptr));
                 PyArray_ITER_NEXT(iter);
            }
            auto result = plask::make_shared<PythonDataVector<T,dim>>(data, mesh);
            return py::object(result);
        }

    }

    template <typename T, int dim>
    static py::object makeDataVector(PyArrayObject* arr, shared_ptr<MeshD<dim>> mesh) {
        size_t ndim = PyArray_NDIM(arr);
        size_t last_dim = PyArray_DIMS(arr)[ndim-1];
        if (ndim == 1) {
            if (last_dim == 2 * mesh->size()) return makeDataVectorImpl<Vec<2,T>, dim>(arr, mesh);
            else if (last_dim == 3 * mesh->size()) return makeDataVectorImpl<Vec<3,T>, dim>(arr, mesh);
            else if (last_dim == 4 * mesh->size()) return makeDataVectorImpl<Tensor3<T>, dim>(arr, mesh);
        } else if (ndim == 2 && std::size_t(PyArray_DIMS(arr)[0]) == mesh->size()) {
            if (last_dim == 2) return makeDataVectorImpl<Vec<2,T>, dim>(arr, mesh);
            else if (last_dim == 3) return makeDataVectorImpl<Vec<3,T>, dim>(arr, mesh);
            else if (last_dim == 4) return makeDataVectorImpl<Tensor3<T>, dim>(arr, mesh);
        } else if (ndim == dim+1) {
            if (last_dim == 2) return makeDataVectorImpl<Vec<2,T>, dim>(arr, mesh);
            else if (last_dim == 3) return makeDataVectorImpl<Vec<3,T>, dim>(arr, mesh);
            else if (last_dim == 4) return makeDataVectorImpl<Tensor3<T>, dim>(arr, mesh);
        }
        return makeDataVectorImpl<T, dim>(arr, mesh);
    }

} // namespace  detail

PLASK_PYTHON_API py::object Data(PyObject* obj, py::object omesh) {
    if (!PyArray_Check(obj)) throw TypeError(u8"data needs to be array object");
    PyArrayObject* arr = (PyArrayObject*)obj;

    try {
        shared_ptr<MeshD<2>> mesh = py::extract<shared_ptr<MeshD<2>>>(omesh);

        switch (PyArray_TYPE(arr)) {
            case NPY_DOUBLE: return detail::makeDataVector<double,2>(arr, mesh);
            case NPY_CDOUBLE: return detail::makeDataVector<dcomplex,2>(arr, mesh);
            default: throw TypeError(u8"Array has wrong dtype (only float and complex allowed)");
        }

    } catch (py::error_already_set) { PyErr_Clear(); try {
        shared_ptr<MeshD<3>> mesh = py::extract<shared_ptr<MeshD<3>>>(omesh);

        switch (PyArray_TYPE(arr)) {
            case NPY_DOUBLE: return detail::makeDataVector<double,3>(arr, mesh);
            case NPY_CDOUBLE: return detail::makeDataVector<dcomplex,3>(arr, mesh);
            default: throw TypeError(u8"Array has wrong dtype (only float and complex allowed)");
        }

    } catch (py::error_already_set) {
        throw TypeError(u8"mesh must be a proper mesh object");
    }}

    return py::object();
}

template <typename T, int dim>
static PythonDataVector<T,dim> PythonDataVector__add__(const PythonDataVector<T,dim>& vec1, const PythonDataVector<T,dim>& vec2) {
    if (vec1.mesh != vec2.mesh)
        throw ValueError(u8"You may only add data on the same mesh");
    return PythonDataVector<T,dim>(vec1 + vec2, vec1.mesh);
}

template <typename T, int dim>
static PythonDataVector<T,dim> PythonDataVector__sub__(const PythonDataVector<T,dim>& vec1, const PythonDataVector<T,dim>& vec2) {
    if (vec1.mesh != vec2.mesh)
        throw ValueError(u8"You may only subtract data on the same mesh");
    return PythonDataVector<T,dim>(vec1 + vec2, vec1.mesh);
}

// template <typename T, int dim>
// static void PythonDataVector__iadd__(const PythonDataVector<T,dim>& vec1, const PythonDataVector<T,dim>& vec2) {
//     if (vec1.mesh != vec2.mesh)
//         throw ValueError("You may only add data on the same mesh");
//     vec1 += vec2;
// }
//
// template <typename T, int dim>
// static void PythonDataVector__isub__(const PythonDataVector<T,dim>& vec1, const PythonDataVector<T,dim>& vec2) {
//     if (vec1.mesh != vec2.mesh)
//         throw ValueError("You may only subtract data on the same mesh");
//     vec1 -= vec2;
// }

template <typename T, int dim>
static PythonDataVector<T,dim> PythonDataVector__neg__(const PythonDataVector<T,dim>& vec) {
    return PythonDataVector<T,dim>(-vec, vec.mesh);
}

template <typename T, int dim>
static PythonDataVector<T,dim> PythonDataVector__mul__(const PythonDataVector<T,dim>& vec, typename detail::basetype<T>::type a) {
    return PythonDataVector<T,dim>(vec * a, vec.mesh);
}

template <typename T, int dim>
static PythonDataVector<T,dim> PythonDataVector__div__(const PythonDataVector<T,dim>& vec, typename detail::basetype<T>::type a) {
    return PythonDataVector<T,dim>(vec / a, vec.mesh);
}

// template <typename T, int dim>
// static void PythonDataVector__imul__(const PythonDataVector<T,dim>& vec, typename detail::basetype<T>::type a) {
//     vec *= a;
// }
//
// template <typename T, int dim>
// void PythonDataVector__idiv__(const PythonDataVector<T,dim>& vec, typename detail::basetype<T>::type a) {
//     vec /= a;
// }


template <typename T, int dim>
static PythonDataVector<const decltype(abs(T())),dim> PythonDataVector__abs__(const PythonDataVector<T,dim>& vec) {
    DataVector<decltype(abs(T()))> absvec(vec.size());
    for (size_t i = 0; i != vec.size(); ++i) absvec[i] = abs(vec[i]);
    return PythonDataVector<const decltype(abs(T())),dim>(std::move(absvec), vec.mesh);
}

template <typename T, int dim>
static PythonDataVector<const decltype(real(T())),dim> PythonDataVector_real(const PythonDataVector<T,dim>& vec) {
    DataVector<decltype(real(T()))> revec(vec.size());
    for (size_t i = 0; i != vec.size(); ++i) revec[i] = real(vec[i]);
    return PythonDataVector<const decltype(real(T())),dim>(std::move(revec), vec.mesh);
}

template <typename T, int dim>
static PythonDataVector<const decltype(imag(T())),dim> PythonDataVector_imag(const PythonDataVector<T,dim>& vec) {
    DataVector<decltype(imag(T()))> imvec(vec.size());
    for (size_t i = 0; i != vec.size(); ++i) imvec[i] = imag(vec[i]);
    return PythonDataVector<const decltype(imag(T())),dim>(std::move(imvec), vec.mesh);
}


template <typename T, int dim>
static bool PythonDataVector__eq__(const PythonDataVector<T,dim>& vec1, const PythonDataVector<T,dim>& vec2) {
    if (vec1.mesh != vec2.mesh) return false;
    for (size_t i = 0; i < vec1.size(); ++i)
        if (vec1[i] != vec2[i]) return false;
    return true;
}

template <typename T, int dim>
static inline py::class_<PythonDataVector<const T,dim>, shared_ptr<PythonDataVector<const T,dim>>> register_data_vector_common()
{
    py::class_<PythonDataVector<const T,dim>, shared_ptr<PythonDataVector<const T,dim>>>
    data("_Data", DATA_DOCSTRING, py::no_init);
    data
        .def_readonly("mesh", &PythonDataVector<const T,dim>::mesh,
            u8"The mesh at which the data was obtained.\n\n"

            u8"The sequential points of this mesh always correspond to the sequential points of\n"
            u8"the data. This implies that ``len(data.mesh) == len(data)`` is always True.\n"
         )
        .def("__len__", &PythonDataVector<const T,dim>::size)
        .def("__getitem__", &PythonDataVector_getitem<const T,dim>)
        .def("__getslice__", &PythonDataVector_getslice<const T,dim>)
        .def("__contains__", &PythonDataVector_contains<const T,dim>)
        .def("__iter__", py::range(&PythonDataVector_begin<const T,dim>, &PythonDataVector_end<const T,dim>))
        .def("__array__", &PythonDataVector__array__<const T,dim>, py::arg("dtype")=py::object())
        .def("__eq__", &PythonDataVector__eq__<const T,dim>)
        .add_property("array", &PythonDataVector_Array<const T,dim>,
            u8"Array formatted by the mesh.\n\n"

            u8"This attribute is available only if the :attr:`mesh` is a rectangular one. It\n"
            u8"contains the held data reshaped to match the shape of the mesh (i.e. the first\n"
            u8"dimension is equal the size of the first mesh axis and so on). If the data type\n"
            u8"is :class:`plask.vec` then the array has one additional dimention equal to 2 for\n"
            u8"2D vectors and 3 for 3D vectors. The vector components are stored in this\n"
            u8"dimention.\n\n"

            u8"Example:\n"
            u8"    >>> msh = plask.mesh.Rectangular2D(plask.mesh.Rectilinear([1, 2]),\n"
            u8"    ... plask.mesh.Rectilinear([10, 20]))\n"
            u8"    >>> dat = Data(array([[[1., 0.], [2., 0.]], [[3., 1.], [4., 1.]]]), msh)\n"
            u8"    >>> dat.array[:,:,0]\n"
            u8"    array([[1., 2.],\n"
            u8"           [3., 4.]])\n\n"

            u8"Accessing this field is efficient, as only the numpy array view is created and\n"
            u8"no data is copied in the memory.\n"
         )
        .def("interpolate", dataInterpolate<const T,dim>, (py::arg("mesh"), "interpolation", py::arg("geometry")=py::object()),
            u8"Interpolate data to a different mesh.\n\n"

            u8"This method interpolated data into a different mesh using specified\n"
            u8"interpolation method. This is exactly the same interpolation that is\n"
            u8"usually done by solvers in their providers.\n\n"

            u8"Args:\n"
            u8"    mesh: Mesh to interpolate into.\n"
            u8"    interpolation: Requested interpolation method.\n"
            u8"    geometry: Optional geometry, over which the interpolation is performed.\n"
            u8"Returns:\n"
            u8"    plask._Data: Interpolated data."
        )
        .add_static_property("dtype", &DataVector_dtype<const T,dim>, "Type of the held values.")
    ;
    data.attr("__name__") = "Data";
    data.attr("__module__") = "plask";

    return data;
}


template <typename T, int dim>
static inline typename std::enable_if<detail::isBasicData<T>::value>::type register_data_vector()
{
    auto data = register_data_vector_common<T,dim>();
    data
        .def("__add__", &PythonDataVector__add__<const T,dim>)
        .def("__sub__", &PythonDataVector__sub__<const T,dim>)
        .def("__mul__", &PythonDataVector__mul__<const T,dim>)
        .def("__rmul__", &PythonDataVector__mul__<const T,dim>)
        .def("__div__", &PythonDataVector__div__<const T,dim>)
        .def("__truediv__", &PythonDataVector__div__<const T,dim>)
        .def("__neg__", &PythonDataVector__neg__<const T,dim>)
        .def("__abs__", &PythonDataVector__abs__<const T,dim>)
        .add_property("real", &PythonDataVector_real<const T,dim>)
        .add_property("imag", &PythonDataVector_imag<const T,dim>)
        // .def("__iadd__", &PythonDataVector__iadd__<const T,dim>)
        // .def("__isub__", &PythonDataVector__isub__<const T,dim>)
        // .def("__imul__", &PythonDataVector__imul__<const T,dim>)
        // .def("__idiv__", &PythonDataVector__idiv__<const T,dim>)
        // .def("__itruediv__", &PythonDataVector__idiv__<const T,dim>)
    ;
}

template <typename T, int dim>
static inline typename std::enable_if<!detail::isBasicData<T>::value>::type register_data_vector()
{
    /*auto data = */register_data_vector_common<T,dim>();
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

    register_data_vector<std::vector<double>, dim>();

    py::implicitly_convertible<PythonDataVector<const double, dim>, PythonDataVector<const Tensor2<double>, dim>>();
    py::implicitly_convertible<PythonDataVector<const dcomplex, dim>, PythonDataVector<const Tensor2<dcomplex>, dim>>();

    py::implicitly_convertible<PythonDataVector<const double, dim>, PythonDataVector<const Tensor3<double>, dim>>();
    py::implicitly_convertible<PythonDataVector<const dcomplex, dim>, PythonDataVector<const Tensor3<dcomplex>, dim>>();

    py::implicitly_convertible<PythonDataVector<const Vec<2,double>, dim>, PythonDataVector<const Tensor2<double>, dim>>();
    py::implicitly_convertible<PythonDataVector<const Vec<2,dcomplex>, dim>, PythonDataVector<const Tensor2<dcomplex>, dim>>();

    py::implicitly_convertible<PythonDataVector<const Vec<3,double>, dim>, PythonDataVector<const Tensor3<double>, dim>>();
    py::implicitly_convertible<PythonDataVector<const Vec<3,dcomplex>, dim>, PythonDataVector<const Tensor3<dcomplex>, dim>>();
}

void register_data_vectors() {
    // Initialize numpy
    if (!plask_import_array()) throw(py::error_already_set());

    register_data_vectors_d<2>();
    register_data_vectors_d<3>();

    py::def("Data", &Data, (py::arg("array"), "mesh"), DATA_DOCSTRING);
}


} // namespace python

template <int dim, typename SrcT, typename DstT>
struct __InterpolateMeta__<python::MeshWrap<dim>, SrcT, DstT, 0>
{
    inline static LazyData<typename std::remove_const<DstT>::type> interpolate(
            const shared_ptr<const python::MeshWrap<dim>>& src_mesh, const DataVector<const SrcT>& src_vec,
            const shared_ptr<const MeshD<dim>>& dst_mesh, InterpolationMethod method, const InterpolationFlags& /*flags*/) {
        OmpLockGuard<OmpNestLock> lock(python::python_omp_lock);
        typedef python::PythonDataVector<const DstT, dim> ReturnedType;
        boost::python::object omesh(const_pointer_cast<MeshD<dim>>(dst_mesh));
        auto source = plask::make_shared<python::PythonDataVector<const SrcT, dim>>(
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


template <>
std::vector<double> InterpolationFlags::reflect<std::vector<double>>(int ax, std::vector<double> val) const {
    if (sym[ax] & 14) {
        std::vector<double> result(val);
        for (double& r: result) r = -r;
        return result;
    }
    else return val;
}


namespace python {

#define INTERPOLATE_NEAREST(M) \
    InterpolationAlgorithm<M<dim>, typename std::remove_const<T>::type, typename std::remove_const<T>::type, INTERPOLATION_NEAREST> \
        ::interpolate(src_mesh, self, dst_mesh, flags)

template <typename T, int dim>
static inline typename std::enable_if<!detail::isBasicData<T>::value, PythonDataVector<T,dim>>::type
dataInterpolateImpl(const PythonDataVector<T,dim>& self, shared_ptr<MeshD<dim>> dst_mesh,
                    InterpolationMethod method, const InterpolationFlags& flags)
{
    if (method != INTERPOLATION_NEAREST)
        writelog(LOG_WARNING, u8"Using 'nearest' algorithm for interpolate(dtype={})", str(py::object(detail::dtype<T>())));

    if (auto src_mesh = dynamic_pointer_cast<RectangularMesh<dim>>(self.mesh))
        return PythonDataVector<T,dim>(INTERPOLATE_NEAREST(RectangularMesh), dst_mesh);
    else if (auto src_mesh = dynamic_pointer_cast<MeshWrap<dim>>(self.mesh))
        return PythonDataVector<T,dim>(INTERPOLATE_NEAREST(MeshWrap), dst_mesh);
        // TODO add new mesh types here

    throw NotImplemented(format(u8"interpolate(source mesh type: {}, interpolation method: {})",
                                typeid(*self.mesh).name(), interpolationMethodNames[method]));
}

template <typename T, int dim>
static inline typename std::enable_if<detail::isBasicData<T>::value, PythonDataVector<T,dim>>::type
dataInterpolateImpl(const PythonDataVector<T,dim>& self, shared_ptr<MeshD<dim>> dst_mesh,
                    InterpolationMethod method, const InterpolationFlags& flags)
{

    if (self.mesh_changed) throw Exception(u8"Cannot interpolate, mesh changed since data retrieval");

    if (auto src_mesh = dynamic_pointer_cast<RectangularMesh<dim>>(self.mesh))
        return PythonDataVector<T,dim>(interpolate(src_mesh, self, dst_mesh, method, flags), dst_mesh);
    else if (auto src_mesh = dynamic_pointer_cast<MeshWrap<dim>>(self.mesh))
        return PythonDataVector<T,dim>(interpolate(src_mesh, self, dst_mesh, method, flags), dst_mesh);
    // TODO add new mesh types here

    throw NotImplemented(format(u8"interpolate(source mesh type: {}, interpolation method: {})",
                                typeid(*self.mesh).name(), interpolationMethodNames[method]));
}

template <typename T, int dim>
PLASK_PYTHON_API PythonDataVector<T,dim> dataInterpolate(const PythonDataVector<T,dim>& self,
                                                       shared_ptr<MeshD<dim>> dst_mesh,
                                                       InterpolationMethod method,
                                                       const py::object& geometry)
{
    InterpolationFlags flags;
    if (geometry != py::object()) {
        py::extract<shared_ptr<const GeometryD<2>>> geometry2d(geometry);
        py::extract<shared_ptr<const GeometryD<3>>> geometry3d(geometry);
        if (geometry2d.check()) {
            flags = InterpolationFlags(geometry2d(),
                                       InterpolationFlags::Symmetry::POSITIVE, InterpolationFlags::Symmetry::POSITIVE);
        } else if (geometry3d.check()) {
            flags = InterpolationFlags(geometry3d(),
                                       InterpolationFlags::Symmetry::POSITIVE, InterpolationFlags::Symmetry::POSITIVE, InterpolationFlags::Symmetry::POSITIVE);
        } else
            throw TypeError(u8"'geometry' argument must be geometry.Geometry instance");
    }

    return dataInterpolateImpl<T, dim>(self, dst_mesh, method, flags);
}


}} // namespace plask::python

#include <plask/data.h>
