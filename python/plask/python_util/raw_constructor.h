#ifndef RAW_CONSTRUCTOR_HPP
#define RAW_CONSTRUCTOR_HPP

#include <boost/python.hpp>
#include <boost/python/detail/api_placeholder.hpp>
#include <boost/python/raw_function.hpp>

namespace plask { namespace python {

namespace detail {

  template <typename F>
  struct raw_constructor_dispatcher
  {
      raw_constructor_dispatcher(F f) : f(boost::python::make_constructor(f)) {}

      PyObject* operator()(PyObject* args, PyObject* keywords)
      {
          boost::python::detail::borrowed_reference_t* ra = boost::python::detail::borrowed_reference(args);
          boost::python::object a(ra);
          return boost::python::incref(
              boost::python::object(
                  f(boost::python::object(a[0]),
                    a,
                    keywords ? boost::python::dict(boost::python::detail::borrowed_reference(keywords)) : boost::python::dict()
                  )
              ).ptr()
          );
      }

   private:
      boost::python::object f;
  };

} // namespace detail

template <typename F>
boost::python::object raw_constructor(F f, std::size_t min_args = 0)
{
    return boost::python::detail::make_raw_function(
        boost::python::objects::py_function(
            detail::raw_constructor_dispatcher<F>(f)
          , boost::mpl::vector2<void, boost::python::object>()
          , int(min_args+1)
          , (std::numeric_limits<unsigned>::max)()
        )
    );
}

}} // namespace plask::python

#endif // RAW_CONSTRUCTOR_HPP
