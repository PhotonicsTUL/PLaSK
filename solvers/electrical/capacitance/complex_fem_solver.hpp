/*
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2023 Lodz University of Technology
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
#ifndef SOLVERS_ELECTRICAL_CAPACITANCE_COMPLEX_FEM_SOLVER_HPP
#define SOLVERS_ELECTRICAL_CAPACITANCE_COMPLEX_FEM_SOLVER_HPP

#include <plask/plask.hpp>
#include <plask/common/fem/fem_solver.hpp>

#include "complex_gauss_matrix.hpp"

namespace plask { namespace electrical { namespace capacitance {

/// Choice of matrix factorization algorithms
enum ComplexFemMatrixAlgorithm {
    ALGORITHM_GAUSS,     ///< Gauss elimination of asymmetric matrix (slower but safer as it uses pivoting)
    ALGORITHM_ITERATIVE  ///< Conjugate gradient iterative solver
};

template <typename SpaceT, typename MeshT> struct ComplexFemSolverWithMesh : public SolverWithMesh<SpaceT, MeshT> {
    ComplexFemMatrixAlgorithm algorithm = ALGORITHM_GAUSS;  ///< Factorization algorithm to use

    ComplexFemSolverWithMesh(const std::string& name = "") : SolverWithMesh<SpaceT, MeshT>(name) {}

    bool parseFemConfiguration(XMLReader& reader, Manager& manager) {
        // if (reader.getNodeName() == "matrix") {
        //     algorithm = reader.enumAttribute<ComplexFemMatrixAlgorithm>("algorithm")
        //                     .value("gauss", ALGORITHM_GAUSS)
        //                     .value("iterative", ALGORITHM_ITERATIVE)
        //                     .get(algorithm);
        //     reader.requireTagEnd();  // </matrix>
        //     return true;
        // }
        return false;
    }

    inline FemMatrix<dcomplex>* getMatrix();
};

template <typename SpaceT, typename MeshT> inline FemMatrix<dcomplex>* ComplexFemSolverWithMesh<SpaceT, MeshT>::getMatrix() {
    switch (algorithm) {
        case ALGORITHM_GAUSS: return new ZgbMatrix(this, this->mesh->size(), this->mesh->minorAxis()->size() + 1);
        // case ALGORITHM_ITERATIVE: return new SparseBandMatrix(this, 2 * this->mesh->size(), 2 * this->mesh->minorAxis()->size());
    }
    return nullptr;
}

template <> inline FemMatrix<dcomplex>* ComplexFemSolverWithMesh<Geometry3D, RectangularMesh<3>>::getMatrix() {
    size_t band = this->mesh->minorAxis()->size() * (this->mesh->mediumAxis()->size() + 1) + 1;
    switch (algorithm) {
        case ALGORITHM_GAUSS: return new ZgbMatrix(this, this->mesh->size(), band);
        // case ALGORITHM_ITERATIVE:
        //     return new SparseBandMatrix(this, 2 * this->mesh->size(), 2 * this->mesh->mediumAxis()->size() * this->mesh->minorAxis()->size(),
        //                                 2 * this->mesh->minorAxis()->size());
    }
    return nullptr;
}

//////////////////////// Solver with masked mesh ////////////////////////

template <typename SpaceT, typename MeshT> struct ComplexFemSolverWithMaskedMesh : public ComplexFemSolverWithMesh<SpaceT, MeshT> {
    static_assert(std::is_base_of<RectangularMesh<MeshT::DIM>, MeshT>::value,
                  "ComplexFemSolverWithMaskedMesh only works with RectangularMesh");

    ComplexFemSolverWithMaskedMesh(const std::string& name = "") : ComplexFemSolverWithMesh<SpaceT, MeshT>(name) {}

  protected:
    plask::shared_ptr<RectangularMaskedMesh<MeshT::DIM>> maskedMesh = plask::make_shared<RectangularMaskedMesh<MeshT::DIM>>();
    EmptyElementsHandling empty_elements = EMPTY_ELEMENTS_DEFAULT;  ///< Should we use full mesh?

  public:
    /// Are we using full mesh?
    EmptyElementsHandling getEmptyElements() const { return empty_elements; }
    /// Set whether we should use full mesh
    void setEmptyElements(EmptyElementsHandling val) {
        empty_elements = val;
        this->invalidate();
    }

    bool parseFemConfiguration(XMLReader& reader, Manager& manager) {
        if (reader.getNodeName() == "mesh") {
            if (reader.hasAttribute("include-empty")) {
                this->writelog(LOG_WARNING, this->getId(), "Attribute 'include-empty' is deprecated, use 'empty-elements' instead");
                empty_elements = reader.requireAttribute<bool>("include-empty") ? EMPTY_ELEMENTS_INCLUDED : EMPTY_ELEMENTS_EXCLUDED;
            }
            empty_elements = reader.enumAttribute<EmptyElementsHandling>("empty-elements")
                                 .value("default", EMPTY_ELEMENTS_DEFAULT)
                                 .value("exclude", EMPTY_ELEMENTS_EXCLUDED)
                                 .value("include", EMPTY_ELEMENTS_INCLUDED)
                                 .value("excluded", EMPTY_ELEMENTS_EXCLUDED)
                                 .value("included", EMPTY_ELEMENTS_INCLUDED)
                                 .get(empty_elements);
            return false;
        }
        return ComplexFemSolverWithMesh<SpaceT, MeshT>::parseFemConfiguration(reader, manager);
    }

    void setupMaskedMesh() {
        if (empty_elements == EMPTY_ELEMENTS_INCLUDED ||
            (this->algorithm == ALGORITHM_ITERATIVE && empty_elements == EMPTY_ELEMENTS_DEFAULT)) {
            maskedMesh->selectAll(*this->mesh);
        } else {
            maskedMesh->reset(*this->mesh, *this->geometry, ~plask::Material::EMPTY);
        }
    }

    void onInitialize() { setupMaskedMesh(); }

    inline FemMatrix<dcomplex>* getMatrix();
};

template <typename SpaceT, typename MeshT> inline FemMatrix<dcomplex>* ComplexFemSolverWithMaskedMesh<SpaceT, MeshT>::getMatrix() {
    size_t band;
    if (empty_elements == EMPTY_ELEMENTS_INCLUDED || this->algorithm == ALGORITHM_ITERATIVE) {
        band = this->mesh->minorAxis()->size() + 1;
    } else {
        band = 0;
        for (auto element : this->maskedMesh->elements()) {
            size_t span = element.getUpUpIndex() - element.getLoLoIndex();
            if (span > band) band = span;
        }
    }
    switch (this->algorithm) {
        case ALGORITHM_GAUSS: return new ZgbMatrix(this, this->maskedMesh->size(), band);
        // case ALGORITHM_ITERATIVE:
        //     if (empty_elements != EMPTY_ELEMENTS_EXCLUDED)
        //         return new SparseBandMatrix(this, this->maskedMesh->size(), this->mesh->minorAxis()->size());
        //     else
        //         return new SparseFreeMatrix(this, this->maskedMesh->size(), this->maskedMesh->elements().size() * 10);
    }
    return nullptr;
}

template <> inline FemMatrix<dcomplex>* ComplexFemSolverWithMaskedMesh<Geometry3D, RectangularMesh<3>>::getMatrix() {
    size_t band;
    if (empty_elements || algorithm == ALGORITHM_ITERATIVE) {
        band = this->mesh->minorAxis()->size() * (this->mesh->mediumAxis()->size() + 1) + 1;
    } else {
        band = 0;
        for (auto element : this->maskedMesh->elements()) {
            size_t span = element.getUpUpUpIndex() - element.getLoLoLoIndex();
            if (span > band) band = span;
        }
    }
    switch (algorithm) {
        case ALGORITHM_GAUSS: return new ZgbMatrix(this, this->maskedMesh->size(), band);
        // case ALGORITHM_ITERATIVE:
        //     if (empty_elements != EMPTY_ELEMENTS_EXCLUDED)
        //         return new SparseBandMatrix(this, this->maskedMesh->size(), mesh->mediumAxis()->size() * mesh->minorAxis()->size(),
        //                                     mesh->minorAxis()->size());
        //     else
        //         return new SparseFreeMatrix(this, this->maskedMesh->size(), this->maskedMesh->elements().size() * 36);
    }
    return nullptr;
}

}}}  // namespace plask::electrical::capacitance

#endif
