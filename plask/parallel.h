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
    class OmpNestLock {
        omp_nest_lock_t lck;

        friend class OmpLockGuard;

      public:

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
    class OmpLockGuard {
        omp_nest_lock_t* lck;

      public:
        OmpLockGuard(): lck(nullptr) {}

        /**
         * Lock the lock.
         * If the lock is already locked by the same thread, just increase its depth
         */
        OmpLockGuard(OmpNestLock& lock): lck(&lock.lck) {
            omp_set_nest_lock(lck);
        }

        /**
         * Move the lock
         */
        OmpLockGuard(OmpLockGuard&& orig): lck(orig.lck){
            orig.lck = nullptr;
        };

        /**
         * Move the lock
         */
        OmpLockGuard& operator=(OmpLockGuard&& orig) {
            if (lck) omp_unset_nest_lock(lck);
            lck = orig.lck;
            orig.lck = nullptr;
            return *this;
        }

        OmpLockGuard(const OmpLockGuard&) = delete;
        OmpLockGuard& operator=(const OmpLockGuard&) = delete;

        /** Unlock the lock */
        ~OmpLockGuard() {
            if (lck) omp_unset_nest_lock(lck);
        }
    };

#else

    // Empty placeholder
    struct OmpNestLock {
        OmpNestLock() = default;
        OmpNestLock(const OmpNestLock&) = delete;
        OmpNestLock& operator=(const OmpNestLock&) = delete;
        OmpNestLock(OmpNestLock&&) = delete;
        OmpNestLock& operator=(OmpNestLock&&) = delete;
    };

    // Empty placeholder
    struct OmpLockGuard {
        OmpLockGuard() {}
        OmpLockGuard(const OmpNestLock&) {}
        OmpLockGuard(OmpLockGuard&&) = default;
        OmpLockGuard& operator=(OmpLockGuard&&) = default;
        OmpLockGuard(const OmpLockGuard&) = delete;
        OmpLockGuard& operator=(const OmpLockGuard&) = delete;
    };

#endif


} // namespace plask

#endif // PLASK__PLASK_PARALLEL_H


