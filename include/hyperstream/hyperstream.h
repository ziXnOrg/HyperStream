#pragma once

/* HyperStream C API (Increment 1)
 * - Pure C ABI with opaque handles
 * - No C++ types leak across the boundary
 * - All functions return status codes; no exceptions cross ABI
 */

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(_WIN64)
#if defined(HYPERSTREAM_C_API_BUILD)
#define HYPERSTREAM_C_API_EXPORT __declspec(dllexport)
#else
#define HYPERSTREAM_C_API_EXPORT __declspec(dllimport)
#endif
#else
#define HYPERSTREAM_C_API_EXPORT __attribute__((visibility("default")))
#endif

#include <stddef.h>
#include <stdint.h>

/* Status codes for all C API functions */
typedef enum hs_status_e {
  HS_OK = 0,
  HS_INVALID_ARG = 1,
  HS_OUT_OF_RANGE = 2,
  HS_CAPACITY_FULL = 3,
  HS_IO_ERROR = 4,
  HS_CORRUPT = 5,
  HS_UNSUPPORTED_CONFIG = 6,
  HS_INTERNAL = 255
} hs_status;

/* Serialization version selector */
typedef enum hs_ser_version_e { HS_SER_V1 = 0, HS_SER_V11_DEFAULT = 1 } hs_ser_version;

/* Opaque handle forward declarations */
typedef struct hs_prototype_mem_s hs_prototype_mem;
typedef struct hs_cluster_mem_s hs_cluster_mem;

/* Creation and destruction */
HYPERSTREAM_C_API_EXPORT hs_status hs_proto_create(uint32_t dim_bits, uint32_t capacity,
                                                   hs_prototype_mem** out);
HYPERSTREAM_C_API_EXPORT void hs_proto_destroy(hs_prototype_mem* m);

HYPERSTREAM_C_API_EXPORT hs_status hs_cluster_create(uint32_t dim_bits, uint32_t capacity,
                                                     hs_cluster_mem** out);
HYPERSTREAM_C_API_EXPORT void hs_cluster_destroy(hs_cluster_mem* m);

/* Introspection */
HYPERSTREAM_C_API_EXPORT uint32_t hs_proto_dim(const hs_prototype_mem* m);
HYPERSTREAM_C_API_EXPORT uint32_t hs_proto_capacity(const hs_prototype_mem* m);
HYPERSTREAM_C_API_EXPORT uint32_t hs_proto_size(const hs_prototype_mem* m);

HYPERSTREAM_C_API_EXPORT uint32_t hs_cluster_dim(const hs_cluster_mem* m);
HYPERSTREAM_C_API_EXPORT uint32_t hs_cluster_capacity(const hs_cluster_mem* m);
HYPERSTREAM_C_API_EXPORT uint32_t hs_cluster_size(const hs_cluster_mem* m);

/* Operations: PrototypeMemory */
HYPERSTREAM_C_API_EXPORT hs_status hs_proto_learn(hs_prototype_mem* m, uint64_t label,
                                                  const uint64_t* words, size_t word_count);
HYPERSTREAM_C_API_EXPORT hs_status hs_proto_classify(const hs_prototype_mem* m,
                                                     const uint64_t* words, size_t word_count,
                                                     uint64_t* out_label);
HYPERSTREAM_C_API_EXPORT hs_status hs_proto_save(const hs_prototype_mem* m, const char* path,
                                                 hs_ser_version ver);
HYPERSTREAM_C_API_EXPORT hs_status hs_proto_load(hs_prototype_mem* m, const char* path);

/* Operations: ClusterMemory */
HYPERSTREAM_C_API_EXPORT hs_status hs_cluster_update(hs_cluster_mem* m, uint64_t label,
                                                     const uint64_t* words, size_t word_count);
HYPERSTREAM_C_API_EXPORT hs_status hs_cluster_finalize(const hs_cluster_mem* m, uint64_t label,
                                                       uint64_t* out_words, size_t out_word_count);
HYPERSTREAM_C_API_EXPORT hs_status hs_cluster_save(const hs_cluster_mem* m, const char* path,
                                                   hs_ser_version ver);
HYPERSTREAM_C_API_EXPORT hs_status hs_cluster_load(hs_cluster_mem* m, const char* path);

/* Notes:
 * - words/out_words represent a packed little-endian bit array of length ceil(dim_bits/64).
 * - Callers must provide correctly sized buffers; functions return HS_INVALID_ARG otherwise.
 * - Implementations must be thread-unsafe (matching core); external synchronization is required.
 */

#ifdef __cplusplus
} /* extern "C" */
#endif
