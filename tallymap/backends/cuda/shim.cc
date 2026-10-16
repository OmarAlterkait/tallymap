// tallymap cuda backend: ONE generic XLA FFI handler that launches NVRTC/nvcc-compiled simt kernels.
//
// Python compiles a kernel module (CUBIN) per program and registers its image here under a 64-bit id
// (tallymap_register).  At execution the handler lazily loads the module into the CURRENT CUDA context
// (the one XLA runs the stream in), memsets the first n_zero result buffers, and launches
//   sj_prep  (optional, flags & 1)  identity init of min/max accumulators
//   sj_main                         persistent threads, occupancy-sized grid
//   sj_post  (optional, flags & 2)  scratch accumulator -> output conversion
// on XLA's stream.  Every kernel takes (by-value struct of all buffer pointers: args first, then results; i64 n_items).
// n_items is a RUNTIME argument (one cubin for every N).  flags & 4: grid not capped by n_items (spawn).
//
// Only the CUDA driver API is used, resolved with dlopen("libcuda.so.1"), so this file needs no CUDA
// headers and builds with a plain host C++17 compiler against jax.ffi.include_dir().
#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

typedef struct CUstream_st* cudaStream_t;
typedef int CUresult;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUcontext;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;

#define SJ_MAXP 504
struct SjParams { void* p[SJ_MAXP]; };

namespace {

struct Drv {
  bool ok = false;
  std::string err;
  CUresult (*CtxGetCurrent)(CUcontext*) = nullptr;
  CUresult (*CtxGetDevice)(CUdevice*) = nullptr;
  CUresult (*ModuleLoadData)(CUmodule*, const void*) = nullptr;
  CUresult (*ModuleGetFunction)(CUfunction*, CUmodule, const char*) = nullptr;
  CUresult (*LaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                           cudaStream_t, void**, void**) = nullptr;
  CUresult (*MemsetD8Async)(CUdeviceptr, unsigned char, size_t, cudaStream_t) = nullptr;
  CUresult (*Occupancy)(int*, CUfunction, int, size_t) = nullptr;
  CUresult (*DeviceGetAttribute)(int*, int, CUdevice) = nullptr;
  CUresult (*GetErrorString)(CUresult, const char**) = nullptr;
  CUresult (*FuncGetAttribute)(int*, int, CUfunction) = nullptr;   // optional: only for the out-of-memory message
};

template <class F>
bool sym(void* h, const char* name, F& f) {
  f = reinterpret_cast<F>(dlsym(h, name));
  return f != nullptr;
}

Drv& drv() {
  static Drv d;
  static std::once_flag once;
  std::call_once(once, [] {
    void* h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!h) { d.err = "dlopen libcuda.so.1 failed"; return; }
    bool ok = sym(h, "cuCtxGetCurrent", d.CtxGetCurrent) && sym(h, "cuCtxGetDevice", d.CtxGetDevice) &&
              sym(h, "cuModuleLoadData", d.ModuleLoadData) && sym(h, "cuModuleGetFunction", d.ModuleGetFunction) &&
              sym(h, "cuLaunchKernel", d.LaunchKernel) && sym(h, "cuMemsetD8Async", d.MemsetD8Async) &&
              sym(h, "cuOccupancyMaxActiveBlocksPerMultiprocessor", d.Occupancy) &&
              sym(h, "cuDeviceGetAttribute", d.DeviceGetAttribute) && sym(h, "cuGetErrorString", d.GetErrorString);
    if (!ok) { d.err = "missing CUDA driver symbol"; return; }
    sym(h, "cuFuncGetAttribute", d.FuncGetAttribute);
    d.ok = true;
  });
  return d;
}

std::string cuerr(CUresult r) {
  const char* s = nullptr;
  if (drv().GetErrorString) drv().GetErrorString(r, &s);
  return std::string("CUDA driver error ") + std::to_string(r) + ": " + (s ? s : "?");
}

struct Loaded {
  CUmodule mod = nullptr;
  CUfunction main = nullptr, prep = nullptr, post = nullptr;
  int sms = 0, occ = 0;
};

// A launch that fails with CUDA_ERROR_OUT_OF_MEMORY is, in practice, the per-thread LOCAL memory (large per-item
// arrays, register spills) that the driver reserves for every resident thread, outside XLA's memory pool.  JAX
// preallocates 75% of the GPU by default, so a kernel can fail there that runs with the pool turned off.
// MEASURED on L40S: the forced-dense reverse kernel of tests/reverse.py (82 KB/thread, ~17 GiB at full residency)
// fails under the default preallocation and passes with XLA_PYTHON_CLIENT_PREALLOCATE=false or MEM_FRACTION=.5.
std::string launch_err(const char* what, CUresult r, CUfunction f, int sms) {
  std::string s = std::string(what) + ": " + cuerr(r);
  if (r != 2 /*CUDA_ERROR_OUT_OF_MEMORY*/) return s;
  int local = -1, per_sm = 0;
  CUdevice dev;
  if (drv().FuncGetAttribute) drv().FuncGetAttribute(&local, 3 /*CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES*/, f);
  if (!drv().CtxGetDevice(&dev)) drv().DeviceGetAttribute(&per_sm, 39 /*MAX_THREADS_PER_MULTIPROCESSOR*/, dev);
  s += ". Launching the kernel needs device memory outside JAX's memory pool";
  if (local > 0 && per_sm > 0) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), ": %d bytes of local memory per thread, which the CUDA driver reserves for "
                  "every resident thread (%.1f GiB on this GPU)", local, (double)local * per_sm * sms / (1 << 30));
    s += buf;
  }
  s += ". JAX preallocates 75% of GPU memory by default: set XLA_PYTHON_CLIENT_PREALLOCATE=false, or a lower "
       "XLA_PYTHON_CLIENT_MEM_FRACTION, before JAX starts; or make the per-item state smaller (large per-item "
       "arrays live in local memory)";
  return s;
}

struct Entry {
  std::string image;
  std::unordered_map<CUcontext, Loaded> loaded;
};

std::mutex g_mu;
std::unordered_map<int64_t, Entry> g_reg;

}  // namespace

extern "C" int tallymap_register(int64_t id, const char* data, size_t n) {
  std::lock_guard<std::mutex> lk(g_mu);
  Entry& e = g_reg[id];
  if (e.image.empty()) e.image.assign(data, n);
  return 0;
}

static ffi::Error SjLaunchImpl(cudaStream_t stream, ffi::RemainingArgs args, ffi::RemainingRets rets,
                               int64_t kernel_id, int32_t block, int32_t n_zero, int32_t flags, int32_t grid,
                               int64_t n_items) {
  Drv& d = drv();
  if (!d.ok) return ffi::Error(ffi::ErrorCode::kInternal, "tallymap: " + d.err);
  CUcontext ctx = nullptr;
  CUresult r = d.CtxGetCurrent(&ctx);
  if (r) return ffi::Error(ffi::ErrorCode::kInternal, cuerr(r));
  Loaded L;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_reg.find(kernel_id);
    if (it == g_reg.end())
      return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                        "tallymap: kernel id not registered in this process: " + std::to_string(kernel_id));
    Entry& e = it->second;
    auto lt = e.loaded.find(ctx);
    if (lt == e.loaded.end()) {
      Loaded n;
      r = d.ModuleLoadData(&n.mod, e.image.data());
      if (r) return ffi::Error(ffi::ErrorCode::kInternal, "cuModuleLoadData: " + cuerr(r));
      r = d.ModuleGetFunction(&n.main, n.mod, "sj_main");
      if (r) return ffi::Error(ffi::ErrorCode::kInternal, "sj_main: " + cuerr(r));
      if (flags & 1) { r = d.ModuleGetFunction(&n.prep, n.mod, "sj_prep"); if (r) return ffi::Error(ffi::ErrorCode::kInternal, cuerr(r)); }
      if (flags & 2) { r = d.ModuleGetFunction(&n.post, n.mod, "sj_post"); if (r) return ffi::Error(ffi::ErrorCode::kInternal, cuerr(r)); }
      CUdevice dev;
      d.CtxGetDevice(&dev);
      d.DeviceGetAttribute(&n.sms, 16 /*CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT*/, dev);
      d.Occupancy(&n.occ, n.main, block, 0);
      if (n.occ < 1) n.occ = 1;
      lt = e.loaded.emplace(ctx, n).first;
    }
    L = lt->second;
  }
  SjParams P;
  std::memset(&P, 0, sizeof(P));
  size_t np = 0;
  const size_t na = args.size(), nr = rets.size();
  if (na + nr > SJ_MAXP) return ffi::Error(ffi::ErrorCode::kInvalidArgument, "tallymap: too many buffers");
  for (size_t i = 0; i < na; ++i) {
    auto b = args.get<ffi::AnyBuffer>(i);
    if (!b.has_value()) return b.error();
    P.p[np++] = b->untyped_data();
  }
  for (size_t i = 0; i < nr; ++i) {
    auto b = rets.get<ffi::AnyBuffer>(i);
    if (!b.has_value()) return b.error();
    void* ptr = (*b)->untyped_data();
    P.p[np++] = ptr;
    if ((int)i < n_zero && (*b)->size_bytes() > 0) {
      r = d.MemsetD8Async((CUdeviceptr)ptr, 0, (*b)->size_bytes(), stream);
      if (r) return ffi::Error(ffi::ErrorCode::kInternal, "memset: " + cuerr(r));
    }
  }
  if (n_items <= 0) return ffi::Error::Success();
  long long nn = n_items;
  void* kp[] = {&P, &nn};
  const int aux_grid = L.sms * 4;
  if (flags & 1) {
    r = d.LaunchKernel(L.prep, aux_grid, 1, 1, 256, 1, 1, 0, stream, kp, nullptr);
    if (r) return ffi::Error(ffi::ErrorCode::kInternal, launch_err("sj_prep", r, L.prep, L.sms));
  }
  long long g = grid > 0 ? grid : (long long)L.sms * (grid < 0 && -grid < L.occ ? -grid : L.occ);   // grid < 0: cap blocks/SM (instrument)
  long long need = (n_items + block - 1) / block;
  if (g > need && !(flags & 4)) g = need;
  if (g < 1) g = 1;
  r = d.LaunchKernel(L.main, (unsigned)g, 1, 1, block, 1, 1, 0, stream, kp, nullptr);
  if (r) return ffi::Error(ffi::ErrorCode::kInternal, launch_err("sj_main", r, L.main, L.sms));
  if (flags & 2) {
    r = d.LaunchKernel(L.post, aux_grid, 1, 1, 256, 1, 1, 0, stream, kp, nullptr);
    if (r) return ffi::Error(ffi::ErrorCode::kInternal, launch_err("sj_post", r, L.post, L.sms));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(TallymapLaunch, SjLaunchImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .RemainingArgs()
                                  .RemainingRets()
                                  .Attr<int64_t>("kernel_id")
                                  .Attr<int32_t>("block")
                                  .Attr<int32_t>("n_zero")
                                  .Attr<int32_t>("flags")
                                  .Attr<int32_t>("grid")
                                  .Attr<int64_t>("n_items"));
