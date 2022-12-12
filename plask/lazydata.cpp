/* 
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#include "lazydata.hpp"

namespace plask {

#ifdef _MSC_VER // MSVC require this while MingW does not accept

#define TEMPLATE_CLASS_FOR_LAZY_DATA(...) \
    template class PLASK_API LazyData< __VA_ARGS__ >; \
    template struct PLASK_API LazyDataImpl< __VA_ARGS__ >;

TEMPLATE_CLASS_FOR_LAZY_DATA(Tensor3< complex<double> >)
TEMPLATE_CLASS_FOR_LAZY_DATA(Tensor2< double >)
TEMPLATE_CLASS_FOR_LAZY_DATA(Vec<3, complex<double>>)
TEMPLATE_CLASS_FOR_LAZY_DATA(Vec<3, double>)
TEMPLATE_CLASS_FOR_LAZY_DATA(Vec<2, double>)
TEMPLATE_CLASS_FOR_LAZY_DATA(double)

#endif

}   // namespace plask
