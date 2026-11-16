/* Mark the code a harness measures, so MIMIR times and weighs that alone (C and C++).

       #include "mimir_region.h"
       make_inputs();             // not measured
       mimir_region_begin();
       solve();                   // measured: time_s, region_peak_rss_mb, region_extra_mb
       mimir_region_end();
       check();                   // not measured

   Outside MIMIR (no MIMIR_REGION), both do nothing. Several regions in turn add their
   times. Header only: copy it next to the harness.

   A GPU's work is asynchronous: each mark first waits for the GPU (CUDA or HIP, when
   the process has loaded it, and OpenACC when compiled with it), so the region holds
   the kernels it launched; under CUDA, the region is also the profiler's range
   (cuProfilerStart/Stop), which Nsight records alone. Nothing to link: the GPU's
   libraries are found among those the process already loaded. */
#ifndef MIMIR_REGION_H
#define MIMIR_REGION_H

#include <dlfcn.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifdef _OPENACC
#include <openacc.h>
#endif

/* Weak: a harness linked without libdl (glibc before 2.34) leaves them null, and the
   GPU is then not looked for. */
#pragma weak dlopen
#pragma weak dlsym

static int mimir_region_fd_ = -2, mimir_region_ack_ = -1;

typedef int (*mimir_region_fn0_)(void);
typedef int (*mimir_region_fn1_)(void *);
typedef int (*mimir_region_fn2_)(void *, int);
typedef int (*mimir_region_fn3_)(int, unsigned *, int *);
static int mimir_region_gpu_ = -1; /* -1 not looked for, 0 none, 1 CUDA, 2 HIP */
static mimir_region_fn0_ mimir_cu_sync_, mimir_cu_start_, mimir_cu_stop_, mimir_hip_sync_;
static mimir_region_fn1_ mimir_cu_count_, mimir_cu_push_, mimir_cu_pop_;
static mimir_region_fn2_ mimir_cu_get_, mimir_cu_retain_;
static mimir_region_fn3_ mimir_cu_state_;
static int (*mimir_cu_release_)(int);

static inline void *mimir_region_loaded_(const char *const *names) {
    for (; *names; ++names) {
        void *h = dlopen(*names, RTLD_NOW | RTLD_NOLOAD);
        if (h) return h;
    }
    return 0;
}

#define MIMIR_REGION_SYM_(h, var, name) (*(void **)(&(var)) = dlsym((h), (name)))

static inline void mimir_region_find_gpu_(void) {
    static const char *const cuda[] = {"libcuda.so.1", "libcuda.so", 0};
    static const char *const hip[] = {"libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5", 0};
    void *h;
    mimir_region_gpu_ = 0;
    if (!dlopen || !dlsym) return;
    if ((h = mimir_region_loaded_(cuda))) {
        MIMIR_REGION_SYM_(h, mimir_cu_sync_, "cuCtxSynchronize");
        MIMIR_REGION_SYM_(h, mimir_cu_start_, "cuProfilerStart");
        MIMIR_REGION_SYM_(h, mimir_cu_stop_, "cuProfilerStop");
        MIMIR_REGION_SYM_(h, mimir_cu_count_, "cuDeviceGetCount");
        MIMIR_REGION_SYM_(h, mimir_cu_get_, "cuDeviceGet");
        MIMIR_REGION_SYM_(h, mimir_cu_state_, "cuDevicePrimaryCtxGetState");
        MIMIR_REGION_SYM_(h, mimir_cu_retain_, "cuDevicePrimaryCtxRetain");
        if (!MIMIR_REGION_SYM_(h, mimir_cu_release_, "cuDevicePrimaryCtxRelease_v2"))
            MIMIR_REGION_SYM_(h, mimir_cu_release_, "cuDevicePrimaryCtxRelease");
        MIMIR_REGION_SYM_(h, mimir_cu_push_, "cuCtxPushCurrent_v2");
        MIMIR_REGION_SYM_(h, mimir_cu_pop_, "cuCtxPopCurrent_v2");
        mimir_region_gpu_ = mimir_cu_sync_ ? 1 : 0;
    } else if ((h = mimir_region_loaded_(hip))) {
        MIMIR_REGION_SYM_(h, mimir_hip_sync_, "hipDeviceSynchronize");
        mimir_region_gpu_ = mimir_hip_sync_ ? 2 : 0;
    }
}

/* Wait for every GPU the process uses: this thread's context, then every active
   primary context (kernels another thread launched); a device never used is left. */
static inline void mimir_region_wait_gpu_(void) {
#ifdef _OPENACC
    acc_wait_all();
#endif
    if (mimir_region_gpu_ < 0) mimir_region_find_gpu_();
    if (mimir_region_gpu_ == 2) {
        mimir_hip_sync_();
    } else if (mimir_region_gpu_ == 1) {
        int n = 0, d, dev;
        mimir_cu_sync_();
        if (mimir_cu_count_ && mimir_cu_get_ && mimir_cu_state_ && mimir_cu_retain_ && mimir_cu_release_ &&
            mimir_cu_push_ && mimir_cu_pop_ && mimir_cu_count_(&n) == 0) {
            for (d = 0; d < n; ++d) {
                unsigned flags;
                int active = 0;
                void *ctx = 0;
                if (mimir_cu_get_(&dev, d) || mimir_cu_state_(dev, &flags, &active) || !active) continue;
                if (mimir_cu_retain_(&ctx, dev)) continue;
                if (mimir_cu_push_(ctx) == 0) {
                    mimir_cu_sync_();
                    mimir_cu_pop_(&ctx);
                }
                mimir_cu_release_(dev);
            }
        }
    }
}

static inline void mimir_region_mark_(const char *word) {
    if (mimir_region_fd_ == -2) {
        const char *path = getenv("MIMIR_REGION");
        char ack[4096];
        mimir_region_fd_ = -1;
        if (path && strlen(path) + 5 < sizeof ack) {
            /* Non-blocking open: with no reader (a rank on another node), give up at once. */
            int fd = open(path, O_WRONLY | O_NONBLOCK);
            strcpy(ack, path);
            strcat(ack, ".ack");
            int fa = fd >= 0 ? open(ack, O_RDONLY | O_NONBLOCK) : -1;
            if (fa >= 0) {
                fcntl(fa, F_SETFL, fcntl(fa, F_GETFL) & ~O_NONBLOCK);
                mimir_region_fd_ = fd;
                mimir_region_ack_ = fa;
            } else if (fd >= 0) {
                close(fd);
            }
        }
    }
    if (mimir_region_fd_ >= 0) {
        char c;
        if (write(mimir_region_fd_, word, strlen(word)) > 0)
            (void)!read(mimir_region_ack_, &c, 1);
    }
}

static inline void mimir_region_begin(void) {
    if (!getenv("MIMIR_REGION")) return;
    mimir_region_wait_gpu_();
    mimir_region_mark_("begin\n");
    if (mimir_region_gpu_ == 1 && mimir_cu_start_) mimir_cu_start_();
}

static inline void mimir_region_end(void) {
    if (!getenv("MIMIR_REGION")) return;
    mimir_region_wait_gpu_();
    if (mimir_region_gpu_ == 1 && mimir_cu_stop_) mimir_cu_stop_();
    mimir_region_mark_("end\n");
}

#endif
