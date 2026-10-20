// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#ifndef AFD_CAM_ASYNC_A5_TRANSPORT_H
#define AFD_CAM_ASYNC_A5_TRANSPORT_H
#include <stdint.h>

// The installed CANN 9.0 HCOMM uses libstdc++ ABI0, whereas torch 2.8 uses
// ABI1. Only these C/POD types cross the separately compiled library boundary.
#ifdef __cplusplus
extern "C" {
#endif
typedef struct AfdA5Transport AfdA5Transport;
typedef struct {
    uint64_t base;
    uint64_t bytes;
    uint64_t token;
} AfdA5Span;
typedef struct {
    uint64_t window_bytes;
    uint32_t rank;
    uint32_t world_size;
} AfdA5GroupInfo;

int afd_a5_query_group(const char *group, AfdA5GroupInfo *info);
int afd_a5_register_window(const char *group, uint64_t base, uint64_t bytes,
    void **handle, uint64_t *token);
int afd_a5_transport_create(const char *group, uint32_t attention_ranks,
    uint32_t experts_per_rank, void *window_handle, const AfdA5Span *windows, uint32_t window_count,
    AfdA5Span source, AfdA5Transport **transport);
int afd_a5_transport_launch(AfdA5Transport *transport, void *stream,
    uint64_t descriptors, uint64_t status);
// Caller first drains all streams and destroys the communication domain while
// its registration creators are alive. This releases only the owned C++ state.
void afd_a5_transport_destroy(AfdA5Transport *transport);
#ifdef __cplusplus
}
#endif
#endif // AFD_CAM_ASYNC_A5_TRANSPORT_H
