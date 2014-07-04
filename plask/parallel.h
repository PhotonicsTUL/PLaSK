#ifndef PLASK__PLASK_PARALLEL_H
#define PLASK__PLASK_PARALLEL_H

#ifdef OPENMP_FOUND
#   include <omp.h>
#endif

namespace plask {

#if defined(OPENMP_FOUND) || defined(DOXYGEN)

    /**
     * OMP nest lock class.
     */
    struct OmpNestLock {
        omp_nest_lock_t lck;

        OmpNestLock(const OmpNestLock&) = delete;
        OmpNestLock& operator=(const OmpNestLock&) = delete;
        OmpNestLock(OmpNestLock&&) = delete;
        OmpNestLock& operator=(OmpNestLock&&) = delete;

        /// Initialize the lock
        OmpNestLock() {
            omp_init_nest_lock(&lck);
        }

        /// Destroy the
        ~OmpNestLock() {
            omp_destroy_nest_lock(&lck);
        }
    };

    /**
     * OMP nest lock guard class.
     */
    struct OmpLockGuard {
        omp_nest_lock_t* lck;

        OmpLockGuard() = delete;
        OmpLockGuard(const OmpLockGuard&) = delete;
        OmpNestLock& operator=(const OmpLockGuard&) = delete;
        OmpLockGuard(OmpLockGuard&&) = delete;
        OmpNestLock& operator=(OmpLockGuard&&) = delete;

        /** Lock the lock.
         * If the lock is already locked by the same thread, just increase its depth
         */
        OmpLockGuard(OmpNestLock& lock): lck(&lock.lck) {
            omp_set_nest_lock(lck);
        }

        /** Unlock the lock */
        ~OmpLockGuard() {
            omp_unset_nest_lock(lck);
        }
    };

#else

    // Empty placeholder
    struct OmpNestLock {
        OmpNestLock(const OmpNestLock&) = delete;
        OmpNestLock& operator=(const OmpNestLock&) = delete;
        OmpNestLock(OmpNestLock&&) = delete;
        OmpNestLock& operator=(OmpNestLock&&) = delete;

        OmpNestLock() = default;
    };

    // Empty placeholder
    struct OmpLockGuard {

        OmpLockGuard() = delete;
        OmpLockGuard(const OmpLockGuard&) = delete;
        OmpNestLock& operator=(const OmpLockGuard&) = delete;
        OmpLockGuard(OmpLockGuard&&) = delete;
        OmpNestLock& operator=(OmpLockGuard&&) = delete;

        OmpLockGuard(const OmpNestLock&) {}
    };

#endif


} // namespace plask

#endif // PLASK__PLASK_PARALLEL_H


