#ifndef PLASK__PLASK_PARALLEL_H
#define PLASK__PLASK_PARALLEL_H

#ifdef OPENMP_FOUND
#   include <omp.h>
#endif

namespace plask {

#if defined(OPENMP_FOUND) || defined(DOXYGEN)

    /// OMP nest lock class.
    class OmpNestLock {

        omp_nest_lock_t lck;

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

        void lock() { omp_set_nest_lock(&lck); }

        void unlock() { omp_unset_nest_lock(&lck); }
    };

    /// OMP lock class.
    class OmpLock {

        omp_lock_t lck;

      public:

        OmpLock(const OmpLock&) = delete;
        OmpLock& operator=(const OmpLock&) = delete;
        OmpLock(OmpLock&&) = delete;
        OmpLock& operator=(OmpLock&&) = delete;

        /// Initialize the lock
        OmpLock() {
            omp_init_lock(&lck);
        }

        /// Destroy the
        ~OmpLock() {
            omp_destroy_lock(&lck);
        }

        void lock() { omp_set_lock(&lck); }

        void unlock() { omp_unset_lock(&lck); }
    };


    /**
     * Template of OMP lock guard class.
     * @tpatam LockType type of lock, either OmpLock or OmpNestLock
     */
    template <typename LockType>
    class OmpLockGuard {

        LockType* lck;

      public:
        OmpLockGuard(): lck(nullptr) {}

        /**
         * Lock the lock.
         * In case of OmpNestLock, if the lock is already locked by the same thread, just increase its depth.
         */
        OmpLockGuard(LockType& lock): lck(&lock) {
            lck->lock();
        }

        /// Move the lock
        OmpLockGuard(OmpLockGuard<LockType>&& orig): lck(orig.lck) {
            orig.lck = nullptr;
        };

        /// Move the lock
        OmpLockGuard& operator=(OmpLockGuard&& orig) {
            if (lck) lck->unlock();
            lck = orig.lck;
            orig.lck = nullptr;
            return *this;
        }

        OmpLockGuard(const OmpLockGuard&) = delete;
        OmpLockGuard& operator=(const OmpLockGuard&) = delete;

        /// Unlock the lock
        ~OmpLockGuard() {
            if (lck) lck->unlock();
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

        void lock() {}
        void unlock() {}
    };

    // Empty placeholder
    struct OmpLock {
        OmpLock() = default;
        OmpLock(const OmpLock&) = delete;
        OmpLock& operator=(const OmpLock&) = delete;
        OmpLock(OmpNestLock&&) = delete;
        OmpLock& operator=(OmpLock&&) = delete;

        void lock() {}
        void unlock() {}
    };

    // Empty placeholder
    template <typename LockType>
    struct OmpLockGuard {
        OmpLockGuard() {}
        OmpLockGuard(const LockType&) {}
        ~OmpLockGuard() {}
        OmpLockGuard(OmpLockGuard<LockType>&&) = default;
        OmpLockGuard& operator=(OmpLockGuard<LockType>&&) = default;
        OmpLockGuard(const OmpLockGuard<LockType>&) = delete;
        OmpLockGuard& operator=(const OmpLockGuard<LockType>&) = delete;
    };

#endif


} // namespace plask

#endif // PLASK__PLASK_PARALLEL_H


