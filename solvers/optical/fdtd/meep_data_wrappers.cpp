#include "meep_data_wrappers.hpp"

namespace plask { namespace solvers { namespace optical_fdtd {

SourceMEEP::SourceMEEP(std::string sType,
                       double ax,
                       double ay,
                       double bx,
                       double by,
                       double lam,
                       double start_time,
                       std::string end_time,
                       double amplitude,
                       double width,
                       double slowness,
                       std::string comp)
    : source_type(sType),
      ax(ax),
      ay(ay),
      bx(bx),
      by(by),
      lam(lam),
      start_time(start_time),
      slowness(slowness),
      amplitude(amplitude),
      width(width),
      component(comp),
      end_time(end_time) {}

}}}