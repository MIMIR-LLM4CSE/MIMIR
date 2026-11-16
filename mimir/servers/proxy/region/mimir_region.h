/* Mark the code a harness measures, so MIMIR times and weighs that alone (C and C++).

       #include "mimir_region.h"
       make_inputs();             // not measured
       mimir_region_begin();
       solve();                   // measured: time_s, region_peak_rss_mb, region_extra_mb
       mimir_region_end();
       check();                   // not measured

   Outside MIMIR (no MIMIR_REGION), both do nothing. Several regions in turn add their
   times. Header only: copy it next to the harness. */
#ifndef MIMIR_REGION_H
#define MIMIR_REGION_H

#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int mimir_region_fd_ = -2, mimir_region_ack_ = -1;

static void mimir_region_mark_(const char *word) {
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

static inline void mimir_region_begin(void) { mimir_region_mark_("begin\n"); }
static inline void mimir_region_end(void) { mimir_region_mark_("end\n"); }

#endif
