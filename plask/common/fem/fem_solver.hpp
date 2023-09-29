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
#ifndef PLASK_COMMON_FEM_FEM_SOLVER_HPP
#define PLASK_COMMON_FEM_FEM_SOLVER_HPP

#include <plask/plask.hpp>
#include "cholesky_matrix.hpp"
#include "gauss_matrix.hpp"
#include "iterative_matrix.hpp"

namespace plask {

/// Choice of matrix factorization algorithms
enum FemMatrixAlgorithm {
    ALGORITHM_CHOLESKY,  ///< Cholesky factorization
    ALGORITHM_GAUSS,     ///< Gauss elimination of asymmetric matrix (slower but safer as it uses pivoting)
    ALGORITHM_ITERATIVE  ///< Conjugate gradient iterative solver
};

template <typename SpaceT, typename MeshT> struct FemSolverWithMesh : public SolverWithMesh<SpaceT, MeshT> {
    FemMatrixAlgorithm algorithm = ALGORITHM_CHOLESKY;  ///< Factorization algorithm to use

    IterativeMatrixParams iter_params;  ///< Parameters of iterative solver

    FemSolverWithMesh(const std::string& name = "") : SolverWithMesh<SpaceT, MeshT>(name) {}

    bool parseFemConfiguration(XMLReader& reader, Manager& manager) {
        if (reader.getNodeName() == "matrix") {
            algorithm = reader.enumAttribute<FemMatrixAlgorithm>("algorithm")
                            .value("cholesky", ALGORITHM_CHOLESKY)
                            .value("gauss", ALGORITHM_GAUSS)
                            .value("iterative", ALGORITHM_ITERATIVE)
                            .get(algorithm);

            if (reader.requireTagOrEnd("iterative")) {
                iter_params.accelerator = reader.enumAttribute<IterativeMatrixParams::Accelelator>("accelerator")
                                              .value("cg", IterativeMatrixParams::ACCEL_CG)
                                              .value("si", IterativeMatrixParams::ACCEL_SI)
                                              .value("sor", IterativeMatrixParams::ACCEL_SOR)
                                              .value("srcg", IterativeMatrixParams::ACCEL_SRCG)
                                              .value("srsi", IterativeMatrixParams::ACCEL_SRSI)
                                              .value("basic", IterativeMatrixParams::ACCEL_BASIC)
                                              .value("me", IterativeMatrixParams::ACCEL_ME)
                                              .value("cgnr", IterativeMatrixParams::ACCEL_CGNR)
                                              .value("lsqr", IterativeMatrixParams::ACCEL_LSQR)
                                              .value("odir", IterativeMatrixParams::ACCEL_ODIR)
                                              .value("omin", IterativeMatrixParams::ACCEL_OMIN)
                                              .value("ores", IterativeMatrixParams::ACCEL_ORES)
                                              .value("iom", IterativeMatrixParams::ACCEL_IOM)
                                              .value("gmres", IterativeMatrixParams::ACCEL_GMRES)
                                              .value("usymlq", IterativeMatrixParams::ACCEL_USYMLQ)
                                              .value("usymqr", IterativeMatrixParams::ACCEL_USYMQR)
                                              .value("landir", IterativeMatrixParams::ACCEL_LANDIR)
                                              .value("lanmin", IterativeMatrixParams::ACCEL_LANMIN)
                                              .value("lanres", IterativeMatrixParams::ACCEL_LANRES)
                                              .value("cgcr", IterativeMatrixParams::ACCEL_CGCR)
                                              .value("bcgs", IterativeMatrixParams::ACCEL_BCGS)
                                              .get(iter_params.accelerator);
                iter_params.preconditioner = reader.enumAttribute<IterativeMatrixParams::Preconditioner>("preconditioner")
                                                 .value("rich", IterativeMatrixParams::PRECOND_RICH)
                                                 .value("jac", IterativeMatrixParams::PRECOND_JAC)
                                                 .value("ljac", IterativeMatrixParams::PRECOND_LJAC)
                                                 .value("ljacx", IterativeMatrixParams::PRECOND_LJACX)
                                                 .value("sor", IterativeMatrixParams::PRECOND_SOR)
                                                 .value("ssor", IterativeMatrixParams::PRECOND_SSOR)
                                                 .value("ic", IterativeMatrixParams::PRECOND_IC)
                                                 .value("mic", IterativeMatrixParams::PRECOND_MIC)
                                                 .value("lsp", IterativeMatrixParams::PRECOND_LSP)
                                                 .value("neu", IterativeMatrixParams::PRECOND_NEU)
                                                 .value("lsor", IterativeMatrixParams::PRECOND_LSOR)
                                                 .value("lssor", IterativeMatrixParams::PRECOND_LSSOR)
                                                 .value("llsp", IterativeMatrixParams::PRECOND_LLSP)
                                                 .value("lneu", IterativeMatrixParams::PRECOND_LNEU)
                                                 .value("bic", IterativeMatrixParams::PRECOND_BIC)
                                                 .value("bicx", IterativeMatrixParams::PRECOND_BICX)
                                                 .value("mbic", IterativeMatrixParams::PRECOND_MBIC)
                                                 .value("mbicx", IterativeMatrixParams::PRECOND_MBICX)
                                                 .get(iter_params.preconditioner);
                iter_params.no_convergence_behavior = reader.enumAttribute<IterativeMatrixParams::NoConvergenceBehavior>("noconv")
                                                          .value("error", IterativeMatrixParams::NO_CONVERGENCE_ERROR)
                                                          .value("warning", IterativeMatrixParams::NO_CONVERGENCE_WARNING)
                                                          .value("continue", IterativeMatrixParams::NO_CONVERGENCE_CONTINUE)
                                                          .get(iter_params.no_convergence_behavior);
                iter_params.maxit = reader.getAttribute<int>("maxit", iter_params.maxit);
                iter_params.maxerr = reader.getAttribute<double>("maxerr", iter_params.maxerr);
                iter_params.nfact = reader.getAttribute<int>("nfact", iter_params.nfact);
                iter_params.omega = reader.getAttribute<double>("omega", iter_params.omega);
                iter_params.ndeg = reader.getAttribute<int>("ndeg", iter_params.ndeg);
                iter_params.lvfill = reader.getAttribute<int>("lvfill", iter_params.lvfill);
                iter_params.ltrunc = reader.getAttribute<int>("ltrunc", iter_params.ltrunc);
                iter_params.ns1 = reader.getAttribute<int>("nsave", iter_params.ns1);
                iter_params.ns2 = reader.getAttribute<int>("nrestart", iter_params.ns2);
                reader.requireTagEnd();  // </iterative>
                reader.requireTagEnd();  // </matrix>
            }

            return true;
        }
        return false;
    }

    inline FemMatrix* getMatrix();
};

template <typename SpaceT, typename MeshT> inline FemMatrix* FemSolverWithMesh<SpaceT, MeshT>::getMatrix() {
    switch (algorithm) {
        case ALGORITHM_CHOLESKY: return new DpbMatrix(this, this->mesh->size(), this->mesh->minorAxis()->size() + 1);
        case ALGORITHM_GAUSS: return new DgbMatrix(this, this->mesh->size(), this->mesh->minorAxis()->size() + 1);
        case ALGORITHM_ITERATIVE: return new SparseBandMatrix(this, this->mesh->size(), this->mesh->minorAxis()->size());
    }
    return nullptr;
}

template <> inline FemMatrix* FemSolverWithMesh<Geometry3D, RectangularMesh<3>>::getMatrix() {
    size_t band = this->mesh->minorAxis()->size() * (this->mesh->mediumAxis()->size() + 1) + 1;
    switch (algorithm) {
        case ALGORITHM_CHOLESKY: return new DpbMatrix(this, this->mesh->size(), band);
        case ALGORITHM_GAUSS: return new DgbMatrix(this, this->mesh->size(), band);
        case ALGORITHM_ITERATIVE:
            return new SparseBandMatrix(this, this->mesh->size(), mesh->mediumAxis()->size() * mesh->minorAxis()->size(),
                                        mesh->minorAxis()->size());
    }
    return nullptr;
}

//////////////////////// Solver with masked mesh ////////////////////////

enum EmptyElementsHandling {
    EMPTY_ELEMENTS_DEFAULT,   ///< Use default handling (exclude for Cholesky, include for iterative)
    EMPTY_ELEMENTS_EXCLUDED,  ///< Exclude empty elements from matrix
    EMPTY_ELEMENTS_INCLUDED   ///< Include empty elements in matrix
};

template <typename SpaceT, typename MeshT> struct FemSolverWithMaskedMesh : public FemSolverWithMesh<SpaceT, MeshT> {
    static_assert(std::is_base_of<RectangularMesh<MeshT::DIM>, MeshT>::value,
                  "FemSolverWithMaskedMesh only works with RectangularMesh");

    FemSolverWithMaskedMesh(const std::string& name = "") : FemSolverWithMesh<SpaceT, MeshT>(name) {}

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
        return FemSolverWithMesh<SpaceT, MeshT>::parseFemConfiguration(reader, manager);
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

    inline FemMatrix* getMatrix();
};

template <typename SpaceT, typename MeshT> inline FemMatrix* FemSolverWithMaskedMesh<SpaceT, MeshT>::getMatrix() {
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
        case ALGORITHM_CHOLESKY: return new DpbMatrix(this, this->maskedMesh->size(), band);
        case ALGORITHM_GAUSS: return new DgbMatrix(this, this->maskedMesh->size(), band);
        case ALGORITHM_ITERATIVE:
            if (empty_elements != EMPTY_ELEMENTS_EXCLUDED)
                return new SparseBandMatrix(this, this->maskedMesh->size(), this->mesh->minorAxis()->size());
            else
                return new SparseFreeMatrix(this, this->maskedMesh->size(), this->maskedMesh->elements().size() * 10);
    }
    return nullptr;
}

template <> inline FemMatrix* FemSolverWithMaskedMesh<Geometry3D, RectangularMesh<3>>::getMatrix() {
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
        case ALGORITHM_CHOLESKY: return new DpbMatrix(this, this->maskedMesh->size(), band);
        case ALGORITHM_GAUSS: return new DgbMatrix(this, this->maskedMesh->size(), band);
        case ALGORITHM_ITERATIVE:
            if (empty_elements != EMPTY_ELEMENTS_EXCLUDED)
                return new SparseBandMatrix(this, this->maskedMesh->size(), mesh->mediumAxis()->size() * mesh->minorAxis()->size(),
                                            mesh->minorAxis()->size());
            else
                return new SparseFreeMatrix(this, this->maskedMesh->size(), this->maskedMesh->elements().size() * 36);
    }
    return nullptr;
}

}  // namespace plask

#endif
