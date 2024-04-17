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
#include "../python_globals.hpp"
#include "../python_property.hpp"

#include "plask/properties/optical.hpp"

namespace plask { namespace python {

struct RefractiveIndexComponent {
    typedef boost::python::objects::enum_base base;

    RefractiveIndexComponent();

  private:
    static void* convertible(PyObject* obj);
    static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data);

  public:
    static PyObject* convert(RefractiveIndex::EnumType x);
};

inline RefractiveIndexComponent::RefractiveIndexComponent() {
    boost::python::to_python_converter<RefractiveIndex::EnumType, RefractiveIndexComponent>();
    boost::python::converter::registry::push_back(&convertible, &construct, boost::python::type_id<RefractiveIndex::EnumType>());
}

PyObject* RefractiveIndexComponent::convert(RefractiveIndex::EnumType val) {
    AxisNames* axes = getCurrentAxes();
    std::string ax;

    switch (val) {
        case RefractiveIndex::COMPONENT_VERT: ax = axes->getNameForVert(); break;
        case RefractiveIndex::COMPONENT_LONG: ax = axes->getNameForLong(); break;
        case RefractiveIndex::COMPONENT_TRAN: ax = axes->getNameForTran(); break;
    }

    return boost::python::incref(boost::python::object(ax + ax).ptr());
}

void* RefractiveIndexComponent::convertible(PyObject* obj) { return PyUnicode_Check(obj) ? obj : 0; }

void RefractiveIndexComponent::construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data) {
    AxisNames* axes = getCurrentAxes();

    RefractiveIndex::EnumType val;
    std::string txt = boost::python::extract<std::string>(obj);

    if (txt == "vv" || txt == axes->getNameForVert() + axes->getNameForVert())
        val = RefractiveIndex::COMPONENT_VERT;
    else if (txt == "ll" || txt == axes->getNameForLong() + axes->getNameForLong())
        val = RefractiveIndex::COMPONENT_LONG;
    else if (txt == "tt" || txt == axes->getNameForTran() + axes->getNameForTran())
        val = RefractiveIndex::COMPONENT_TRAN;
    else
        throw ValueError("bad tensor component '{}' (must be '{}', '{}', '{}', 'll', 'tt', or 'vv')", txt,
                         axes->getNameForVert() + axes->getNameForVert(), axes->getNameForLong() + axes->getNameForLong(),
                         axes->getNameForTran() + axes->getNameForTran());

    void* const storage = ((boost::python::converter::rvalue_from_python_storage<RefractiveIndex::EnumType>*)data)->storage.bytes;
    new (storage) RefractiveIndex::EnumType(val);
    data->convertible = storage;
}

void register_standard_properties_refractive(const py::object& flow_module) {
    registerProperty<RefractiveIndex>(flow_module);
    registerProperty<Epsilon>(flow_module);

    RefractiveIndexComponent();
}

}}  // namespace plask::python
