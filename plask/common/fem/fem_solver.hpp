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
    int iterlim = 1000;                                 ///< Allowed residual iteration for iterative method
    double itererr = 1e-6;                              ///< Maximum number of iterations for iterative method

    SparseBandMatrix::Accelelator iter_accelerator = SparseBandMatrix::ACCEL_CG;          ///< Iterative solver accelerator
    SparseBandMatrix::Preconditioner iter_preconditioner = SparseBandMatrix::PRECOND_IC;  ///< Iterative solver preconditioner

    FemSolverWithMesh(const std::string& name = "") : SolverWithMesh<SpaceT, MeshT>(name) {}

    bool parseFemConfiguration(XMLReader& reader, Manager& manager) {
        if (reader.getNodeName() == "matrix") {
            algorithm = reader.enumAttribute<FemMatrixAlgorithm>("algorithm")
                            .value("cholesky", ALGORITHM_CHOLESKY)
                            .value("gauss", ALGORITHM_GAUSS)
                            .value("iterative", ALGORITHM_ITERATIVE)
                            .get(algorithm);
            itererr = reader.getAttribute<double>("itererr", itererr);
            iterlim = reader.getAttribute<size_t>("iterlim", iterlim);

            iter_accelerator = reader.enumAttribute<SparseBandMatrix::Accelelator>("iter-accelerator")
                                   .value("cg", SparseBandMatrix::ACCEL_CG)
                                   .value("si", SparseBandMatrix::ACCEL_SI)
                                   .value("sor", SparseBandMatrix::ACCEL_SOR)
                                   .value("srcg", SparseBandMatrix::ACCEL_SRCG)
                                   .value("srsi", SparseBandMatrix::ACCEL_SRSI)
                                   .value("basic", SparseBandMatrix::ACCEL_BASIC)
                                   .value("me", SparseBandMatrix::ACCEL_ME)
                                   .value("cgnr", SparseBandMatrix::ACCEL_CGNR)
                                   .value("lsqr", SparseBandMatrix::ACCEL_LSQR)
                                   .value("odir", SparseBandMatrix::ACCEL_ODIR)
                                   .value("omin", SparseBandMatrix::ACCEL_OMIN)
                                   .value("ores", SparseBandMatrix::ACCEL_ORES)
                                   .value("iom", SparseBandMatrix::ACCEL_IOM)
                                   .value("gmres", SparseBandMatrix::ACCEL_GMRES)
                                   .value("usymlq", SparseBandMatrix::ACCEL_USYMLQ)
                                   .value("usymqr", SparseBandMatrix::ACCEL_USYMQR)
                                   .value("landir", SparseBandMatrix::ACCEL_LANDIR)
                                   .value("lanmin", SparseBandMatrix::ACCEL_LANMIN)
                                   .value("lanres", SparseBandMatrix::ACCEL_LANRES)
                                   .value("cgcr", SparseBandMatrix::ACCEL_CGCR)
                                   .value("bcgs", SparseBandMatrix::ACCEL_BCGS)
                                   .get(iter_accelerator);

            iter_preconditioner = reader.enumAttribute<SparseBandMatrix::Preconditioner>("iter-preconditioner")
                                      .value("rich", SparseBandMatrix::PRECOND_RICH)
                                      .value("jac", SparseBandMatrix::PRECOND_JAC)
                                      .value("ljac", SparseBandMatrix::PRECOND_LJAC)
                                      .value("ljacx", SparseBandMatrix::PRECOND_LJACX)
                                      .value("sor", SparseBandMatrix::PRECOND_SOR)
                                      .value("ssor", SparseBandMatrix::PRECOND_SSOR)
                                      .value("ic", SparseBandMatrix::PRECOND_IC)
                                      .value("mic", SparseBandMatrix::PRECOND_MIC)
                                      .value("lsp", SparseBandMatrix::PRECOND_LSP)
                                      .value("neu", SparseBandMatrix::PRECOND_NEU)
                                      .value("lsor", SparseBandMatrix::PRECOND_LSOR)
                                      .value("lssor", SparseBandMatrix::PRECOND_LSSOR)
                                      .value("llsp", SparseBandMatrix::PRECOND_LLSP)
                                      .value("lneu", SparseBandMatrix::PRECOND_LNEU)
                                      .value("bic", SparseBandMatrix::PRECOND_BIC)
                                      .value("bicx", SparseBandMatrix::PRECOND_BICX)
                                      .value("mbic", SparseBandMatrix::PRECOND_MBIC)
                                      .value("mbicx", SparseBandMatrix::PRECOND_MBICX)
                                      .get(iter_preconditioner);

            reader.requireTagEnd();
            return true;
        }
        return false;
    }

    inline FemMatrix* getMatrix() const;
};

template <typename SpaceT, typename MeshT> inline FemMatrix* FemSolverWithMesh<SpaceT, MeshT>::getMatrix() const {
    switch (algorithm) {
        case ALGORITHM_CHOLESKY: return new DpbMatrix(this, this->mesh->size(), this->mesh->minorAxis()->size() + 1);
        case ALGORITHM_GAUSS: return new DgbMatrix(this, this->mesh->size(), this->mesh->minorAxis()->size() + 1);
        case ALGORITHM_ITERATIVE: return new SparseBandMatrix(this, this->mesh->size(), this->mesh->minorAxis()->size());
    }
    return nullptr;
}

template <> inline FemMatrix* FemSolverWithMesh<Geometry3D, RectangularMesh<3>>::getMatrix() const {
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

template <typename SpaceT, typename MeshT> struct FemSolverWithMaskedMesh : public FemSolverWithMesh<SpaceT, MeshT> {
    static_assert(std::is_base_of<RectangularMesh<MeshT::DIM>, MeshT>::value,
                  "FemSolverWithMaskedMesh only works with RectangularMesh");

    FemSolverWithMaskedMesh(const std::string& name = "") : FemSolverWithMesh<SpaceT, MeshT>(name) {}

  protected:
    plask::shared_ptr<RectangularMaskedMesh<MeshT::DIM>> maskedMesh = plask::make_shared<RectangularMaskedMesh<MeshT::DIM>>();
    bool use_full_mesh;  ///< Should we use full mesh?

  public:
    /// Are we using full mesh?
    bool usingFullMesh() const { return use_full_mesh; }
    /// Set whether we should use full mesh
    void useFullMesh(bool val) {
        use_full_mesh = val;
        this->invalidate();
    }

    bool parseFemConfiguration(XMLReader& reader, Manager& manager) {
        if (reader.getNodeName() == "mesh") {
            use_full_mesh = reader.getAttribute<bool>("include-empty", use_full_mesh);
            return false;
        }
        return FemSolverWithMesh<SpaceT, MeshT>::parseFemConfiguration(reader, manager);
    }

    void setupMaskedMesh() {
        if (use_full_mesh || this->algorithm == ALGORITHM_ITERATIVE) {
            if (!use_full_mesh) writelog(LOG_WARNING, this->getId(), "For iterative algorithm empty materials are always included");
            maskedMesh->selectAll(*this->mesh);
        } else {
            maskedMesh->reset(*this->mesh, *this->geometry, ~plask::Material::EMPTY);
        }
    }

    void onInitialize() { setupMaskedMesh(); }

    inline FemMatrix* getMatrix() const;
};

template <typename SpaceT, typename MeshT> inline FemMatrix* FemSolverWithMaskedMesh<SpaceT, MeshT>::getMatrix() const {
    size_t band;
    if (use_full_mesh || this->algorithm == ALGORITHM_ITERATIVE) {
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
        case ALGORITHM_ITERATIVE: return new SparseBandMatrix(this, this->maskedMesh->size(), this->mesh->minorAxis()->size());
    }
    return nullptr;
}

template <> inline FemMatrix* FemSolverWithMaskedMesh<Geometry3D, RectangularMesh<3>>::getMatrix() const {
    size_t band;
    if (use_full_mesh || algorithm == ALGORITHM_ITERATIVE) {
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
            return new SparseBandMatrix(this, this->maskedMesh->size(), mesh->mediumAxis()->size() * mesh->minorAxis()->size(),
                                        mesh->minorAxis()->size());
    }
    return nullptr;
}

}  // namespace plask

#endif
