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
#ifndef PLASK__PLASK_PARALLEL_H
#define PLASK__PLASK_PARALLEL_H

#include <plask/config.hpp>

#ifdef OPENMP_FOUND
#    include <omp.h>
#endif

namespace plask {

/// State of the current lock
struct OmpLockState {};

#if defined(OPENMP_FOUND) || defined(DOXYGEN)

/// OMP environment class
struct OmpEnv {
    virtual ~OmpEnv() = default;
    virtual void enable() = 0;
    virtual void disable() = 0;
};

/**
 * OMP enabler class
 *
 * An instance of this class should be created at before the parallel section.
 */
struct OmpEnabler {
    PLASK_API static OmpEnv* env;

    OmpEnabler() {
        if (env) env->enable();
    }

    ~OmpEnabler() {
        if (env) env->disable();
    }

    inline operator bool() const { return true; }  // so that it can be used in an `if` statement
};

#    ifdef _MSC_VER
#        define PLASK_OMP_PARALLEL \
            if (plask::OmpEnabler omp_enabler{}) __pragma(omp parallel)
#        define PLASK_OMP_PARALLEL_FOR \
            if (plask::OmpEnabler omp_enabler{}) __pragma(omp parallel for)
#        define PLASK_OMP_PARALLEL_FOR_1 \
            if (plask::OmpEnabler omp_enabler{}) __pragma(omp parallel for schedule(dynamic,1))
#    else
#        define PLASK_OMP_PARALLEL \
            if (plask::OmpEnabler omp_enabler{}) _Pragma("omp parallel")
#        define PLASK_OMP_PARALLEL_FOR \
            if (plask::OmpEnabler omp_enabler{}) _Pragma("omp parallel for")
#        define PLASK_OMP_PARALLEL_FOR_1 \
            if (plask::OmpEnabler omp_enabler{}) _Pragma("omp parallel for schedule(dynamic,1)")
#    endif

/// Abstract OMP lock
struct OmpLock {
    OmpLock(const OmpLock&) = delete;
    OmpLock& operator=(const OmpLock&) = delete;
    OmpLock(OmpLock&&) = delete;
    OmpLock& operator=(OmpLock&&) = delete;

    OmpLock() {}
    virtual ~OmpLock() {}

    virtual OmpLockState* lock() = 0;
    virtual void unlock(OmpLockState*) = 0;
};

/// OMP nest lock class
class OmpNestedLock : public OmpLock {
    omp_nest_lock_t lck;

  public:
    OmpNestedLock() { omp_init_nest_lock(&lck); }
    ~OmpNestedLock() { omp_destroy_nest_lock(&lck); }
    OmpLockState* lock() override { omp_set_nest_lock(&lck); return nullptr; }
    void unlock(OmpLockState*) override { omp_unset_nest_lock(&lck); }
};

/// OMP lock class
class OmpSingleLock : public OmpLock {
    omp_lock_t lck;

  public:
    OmpSingleLock() { omp_init_lock(&lck); }
    ~OmpSingleLock() { omp_destroy_lock(&lck); }
    OmpLockState* lock() override { omp_set_lock(&lck); return nullptr; }
    void unlock(OmpLockState*) override { omp_unset_lock(&lck); }
};

/// OMP lock guard class
class OmpLockGuard {
    OmpLock* lock;
    OmpLockState* state;

  public:
    OmpLockGuard() : lock(nullptr) {}

    /**
     * Lock the lock.
     * In case of OmpNestedLock, if the lock is already locked by the same thread, just increase its depth.
     */
    OmpLockGuard(OmpLock& lock) : lock(&lock) { state = lock.lock(); }

    /// Move the lock
    OmpLockGuard(OmpLockGuard&& orig) : lock(orig.lock) { orig.lock = nullptr; };

    /// Move the lock
    OmpLockGuard& operator=(OmpLockGuard&& orig) {
        if (lock) lock->unlock(state);
        lock = orig.lock;
        state = orig.state;
        orig.lock = nullptr;
        return *this;
    }

    OmpLockGuard(const OmpLockGuard&) = delete;
    OmpLockGuard& operator=(const OmpLockGuard&) = delete;

    /// Unlock the lock
    ~OmpLockGuard() {
        if (lock) lock->unlock(state);
    }
};

#else

struct OmpEnabler {};

#    define PLASK_OMP_PARALLEL
#    define PLASK_OMP_PARALLEL_FOR
#    define PLASK_OMP_PARALLEL_FOR_1

// Empty placeholder
struct OmpLock {
    OmpLock(const OmpLock&) = delete;
    OmpLock& operator=(const OmpLock&) = delete;
    OmpLock(OmpLock&&) = delete;
    OmpLock& operator=(OmpLock&&) = delete;

    OmpLock() {}

    inline OmpLockState* lock() { return nullptr; }
    inline void unlock(OmpLockState*) {}
};

#    define OmpNestedLock OmpLock
#    define OmpSingleLock OmpLock

// Empty placeholder
struct OmpLockGuard {
    OmpLockGuard() {}
    OmpLockGuard(const OmpLock&) {}
    ~OmpLockGuard() {}
    OmpLockGuard(OmpLockGuard&&) = default;
    OmpLockGuard& operator=(OmpLockGuard&&) = default;
    OmpLockGuard(const OmpLockGuard&) = delete;
    OmpLockGuard& operator=(const OmpLockGuard&) = delete;
};

#endif

}  // namespace plask

#endif  // PLASK__PLASK_PARALLEL_H
