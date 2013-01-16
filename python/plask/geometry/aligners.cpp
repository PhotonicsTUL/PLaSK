#include "geometry.h"
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <boost/algorithm/string.hpp>

#include <plask/geometry/container.h>


namespace plask { namespace python {

namespace detail {

    template <align::Direction... directions>
    struct Aligner_to_Python
    {
        typedef align::Aligner<directions...> AlignerType;

        static PyObject* convert(const AlignerType& aligner) {
            py::dict dict;
            for (auto i: aligner.asDict(config.axes)) {
                dict[i.first] = i.second;
            }
            return py::incref(dict.ptr());
        }
    };

    static void* aligner_convertible(PyObject* obj)
    {
        if (!PyDict_Check(obj)) return 0;
        return obj;
    }

    template <align::Direction... directions>
    static void aligner_construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data)
    {
        typedef align::Aligner<directions...> AlignerType;

        // Grab pointer to memory into which to construct the new Aligner
        void* storage = ((boost::python::converter::rvalue_from_python_storage<AlignerType>*)data)->storage.bytes;

        std::map<std::string, double> map;

        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(obj, &pos, &key, &value)) {
            map[py::extract<std::string>(key)] = py::extract<double>(value);
        }

        auto aligner = new(storage) AlignerType;

        *aligner = align::fromDictionary<directions...>([&](const std::string& name) -> boost::optional<double> {
                                                            boost::optional<double> result;
                                                            auto found = map.find(name);
                                                            if (found != map.end()) {
                                                                result.reset(found->second);
                                                                map.erase(found);
                                                            }
                                                            return result;
                                                        },
                                                        config.axes
                                                       );

        if (!map.empty()) throw TypeError("Got unexpected alignment keyword '%1%'", map.begin()->first);

        // Stash the memory chunk pointer for later use by boost.python
        data->convertible = storage;
    }
}

template <align::Direction... directions>
static inline void register_aligner() {
    py::to_python_converter<align::Aligner<directions...>, detail::Aligner_to_Python<directions...>>();
    py::converter::registry::push_back(&detail::aligner_convertible, &detail::aligner_construct<directions...>, py::type_id<align::Aligner<directions...>>());
}

void register_geometry_aligners()
{
    constexpr Primitive<3>::Direction L = Primitive<3>::DIRECTION_LONG;
    constexpr Primitive<3>::Direction T = Primitive<3>::DIRECTION_TRAN;
    constexpr Primitive<3>::Direction V = Primitive<3>::DIRECTION_VERT;

    register_aligner<L>();
    register_aligner<T>();
    register_aligner<V>();

    register_aligner<L,T>();
    register_aligner<L,V>();
    register_aligner<T,V>();
}





}} // namespace plask::python
