#include "geometry.h"
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <boost/algorithm/string.hpp>

#include <plask/geometry/container.h>


namespace plask { namespace python {

typedef align::AxisAligner<Primitive<3>::DIRECTION_TRAN> A2;
typedef align::AxisAligner<Primitive<3>::DIRECTION_LONG> A2l;
typedef align::Aligner3D<Primitive<3>::DIRECTION_LONG, Primitive<3>::DIRECTION_TRAN> A3;

namespace detail {

    template <typename AlignerT>
    struct Aligner_to_Python {
        static PyObject* convert(const AlignerT& aligner) {
            py::dict dict;
            for (auto i: aligner.asDict(config.axes)) {
                dict[i.first] = i.second;
            }
            return py::incref(dict.ptr());
        }
    };

    static void* aligner_convertible(PyObject* obj) {
        if (!PyDict_Check(obj)) return 0;
        return obj;
    }

    template <typename AlignerT> struct AlignerFromDictionary;

    template <align::Direction direction>
    struct AlignerFromDictionary<align::AxisAligner<direction>> {
        template <typename Dict>
        static align::AxisAligner<direction> get(Dict dict) {
            return align::fromDictionary<direction>(dict, config.axes);
        }
    };

    template <align::Direction direction1, align::Direction direction2>
    struct AlignerFromDictionary<align::Aligner3D<direction1,direction2>> {
        template <typename Dict>
        static align::Aligner3D<direction1,direction2> get(Dict dict) {
            return align::fromDictionary<direction1,direction2>(dict, config.axes);
        }
    };

    template <typename AlignerT>
    static void aligner_construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data) {
        // Grab pointer to memory into which to construct the new Aligner
        void* storage = ((boost::python::converter::rvalue_from_python_storage<AlignerT>*)data)->storage.bytes;

        std::map<std::string, double> map;

        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(obj, &pos, &key, &value)) {
            map[py::extract<std::string>(key)] = py::extract<double>(value);
        }

        auto aligner = new(storage) AlignerT;

        *aligner = AlignerFromDictionary<AlignerT>::get([&](const std::string& name) -> boost::optional<double> {
                                                            boost::optional<double> result;
                                                            auto found = map.find(name);
                                                            if (found != map.end()) {
                                                                result.reset(found->second);
                                                                map.erase(found);
                                                            }
                                                            return result;
                                                        });

        if (!map.empty()) throw KeyError(map.begin()->first);

        // Stash the memory chunk pointer for later use by boost.python
        data->convertible = storage;
    }
}

void register_geometry_aligners()
{
    const Primitive<3>::Direction L = Primitive<3>::DIRECTION_LONG;
    const Primitive<3>::Direction T = Primitive<3>::DIRECTION_TRAN;
    const Primitive<3>::Direction V = Primitive<3>::DIRECTION_VERT;

    py::to_python_converter<align::AxisAligner<L>, detail::Aligner_to_Python<align::AxisAligner<L>>>();
    py::to_python_converter<align::AxisAligner<T>, detail::Aligner_to_Python<align::AxisAligner<T>>>();
    py::to_python_converter<align::AxisAligner<V>, detail::Aligner_to_Python<align::AxisAligner<V>>>();

    py::to_python_converter<align::Aligner3D<L,T>, detail::Aligner_to_Python<align::Aligner3D<L,T>>>();
    py::to_python_converter<align::Aligner3D<L,V>, detail::Aligner_to_Python<align::Aligner3D<L,V>>>();
    py::to_python_converter<align::Aligner3D<T,V>, detail::Aligner_to_Python<align::Aligner3D<T,V>>>();

    py::converter::registry::push_back(&detail::aligner_convertible, &detail::aligner_construct<align::AxisAligner<L>>, py::type_id<align::AxisAligner<L>>());
    py::converter::registry::push_back(&detail::aligner_convertible, &detail::aligner_construct<align::AxisAligner<T>>, py::type_id<align::AxisAligner<T>>());
    py::converter::registry::push_back(&detail::aligner_convertible, &detail::aligner_construct<align::AxisAligner<V>>, py::type_id<align::AxisAligner<V>>());

    py::converter::registry::push_back(&detail::aligner_convertible, &detail::aligner_construct<align::Aligner3D<L,T>>, py::type_id<align::Aligner3D<L,T>>());
    py::converter::registry::push_back(&detail::aligner_convertible, &detail::aligner_construct<align::Aligner3D<L,V>>, py::type_id<align::Aligner3D<L,V>>());
    py::converter::registry::push_back(&detail::aligner_convertible, &detail::aligner_construct<align::Aligner3D<T,V>>, py::type_id<align::Aligner3D<T,V>>());

}





}} // namespace plask::python
