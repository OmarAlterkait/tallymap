// tallymap kernel backend, CPU target: ONE generic XLA FFI handler that runs host-compiled kernels.
//
// Python compiles each program's generated C into a shared object (host.py) that exports
//   extern "C" void sj_host_run(int phase, void** p, int np, long long n_items, int tid, int nthreads)
// (phase 0 = sj_prep, 1 = sj_main, 2 = sj_post) and registers its entry point here under the same 64-bit id the
// cuda target uses (tallymap_host_register).  At execution the handler zeroes the first n_zero result buffers and
// runs the phases on an OpenMP team of T threads, each playing one CUDA thread of a one-thread block
// (blockIdx.x = tid, gridDim.x = T): sj_main claims items from the shared counter exactly as on a GPU.
//
// T = $OMP_NUM_THREADS when set, else the CPUs this process may run on (sched_getaffinity: the cpuset / Slurm
// allocation), capped by n_items unless flags & 4 (spawn: children).  OpenMP, not a hand-rolled pthread team:
// libgomp keeps its threads between launches and wakes them in parallel (measured, rw65k forward at 32 threads on
// an EPYC 7713: 3.1 ms; a thread per launch 6.4 ms, a persistent pthread team with a serial wake 8.0 ms).
//
// Stacks: per-item local arrays live on the stack (the CPU analogue of CUDA's local memory).  The worker threads
// get OMP_STACKSIZE, which host.py sets before libgomp loads.  The thread that calls the handler is XLA's, with a
// stack tallymap does not size: it runs a share of the items only when its free stack holds the kernel's
// -fstack-usage bound (`stack`) with a margin, else the team has one more thread and the caller only waits.
// The kernels themselves include no XLA or OpenMP headers: only this file does, built once per toolchain.
#include <fenv.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

typedef void (*SjHostRun)(int, void**, int, long long, int, int);

namespace {

std::mutex g_mu;
std::unordered_map<int64_t, SjHostRun>& registry() {
  static std::unordered_map<int64_t, SjHostRun> r;
  return r;
}

int default_threads() {
  if (const char* e = std::getenv("OMP_NUM_THREADS")) {
    int v = std::atoi(e);
    if (v > 0) return v;
  }
  cpu_set_t s;
  CPU_ZERO(&s);
  if (sched_getaffinity(0, sizeof(s), &s) == 0) {
    int c = CPU_COUNT(&s);
    if (c > 0) return c;
  }
  return 1;
}

// free stack below this frame on the calling thread, or 0 when it cannot be read
long long caller_free_stack() {
  pthread_attr_t a;
  if (pthread_getattr_np(pthread_self(), &a) != 0) return 0;
  void* lo = nullptr;
  size_t sz = 0;
  const int r = pthread_attr_getstack(&a, &lo, &sz);
  pthread_attr_destroy(&a);
  if (r != 0 || lo == nullptr) return 0;
  char here;
  return (long long)(&here - (char*)lo);
}

constexpr long long CALLER_MARGIN = 256 << 10;   // what the handler and libgomp themselves use below this frame

}  // namespace

extern "C" int tallymap_host_register(int64_t id, void* fn) {
  std::lock_guard<std::mutex> lk(g_mu);
  registry()[id] = reinterpret_cast<SjHostRun>(fn);
  return 1;
}

extern "C" int tallymap_host_threads() { return default_threads(); }

static ffi::Error HostLaunchImpl(ffi::RemainingArgs args, ffi::RemainingRets rets, int64_t kernel_id, int32_t n_zero,
                                 int32_t flags, int64_t n_items, int64_t stack, int32_t n_dev) {
  SjHostRun run = nullptr;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = registry().find(kernel_id);
    if (it != registry().end()) run = it->second;
  }
  if (!run) return ffi::Error(ffi::ErrorCode::kInternal, "tallymap: host kernel not registered in this process");
  std::vector<void*> p;
  p.reserve(args.size() + rets.size());
  for (size_t i = 0; i < args.size(); ++i) {
    auto b = args.get<ffi::AnyBuffer>(i);
    if (!b.has_value()) return b.error();
    p.push_back(b->untyped_data());
  }
  for (size_t i = 0; i < rets.size(); ++i) {
    auto b = rets.get<ffi::AnyBuffer>(i);
    if (!b.has_value()) return b.error();
    void* ptr = (*b)->untyped_data();
    if ((int)i < n_zero && (*b)->size_bytes() > 0) std::memset(ptr, 0, (*b)->size_bytes());
    p.push_back(ptr);
  }
  if (n_items <= 0) return ffi::Error::Success();
  // n_dev: the devices of this computation that run it at once (a sharded jit): they share the process's CPUs
  int nt = default_threads() / (n_dev > 1 ? n_dev : 1);
  if (nt < 1) nt = 1;
  if (!(flags & 4) && (long long)nt > n_items) nt = (int)n_items;
  // the caller works only when its own stack holds the kernel's frames; otherwise it is an extra, idle member
  const int idle = caller_free_stack() >= stack + CALLER_MARGIN ? 0 : 1;
  void** const pp = p.data();
  const int np = (int)p.size();
  const long long nn = n_items;
#pragma omp parallel num_threads(nt + idle)
  {
    // IEEE: libgomp's threads, like XLA:CPU's, may run with subnormals flushed to zero (measured: 1e-45f + 1e-45f
    // gave 0 in the kernel, as in XLA:CPU; IEEE gives 3e-45)
    fenv_t saved;
    fegetenv(&saved);
    fesetenv(FE_DFL_ENV);
    const int off = omp_get_num_threads() > 1 ? idle : 0;   // (the runtime may grant fewer threads than asked)
    const int me = omp_get_thread_num() - off;
    const int team = omp_get_num_threads() - off;
    if (flags & 1) {
      if (me >= 0) run(0, pp, np, nn, me, team);
#pragma omp barrier
    }
    if (me >= 0) run(1, pp, np, nn, me, team);
    if (flags & 2) {
#pragma omp barrier
      if (me >= 0) run(2, pp, np, nn, me, team);
    }
    fesetenv(&saved);
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(TallymapHostLaunch, HostLaunchImpl,
                              ffi::Ffi::Bind()
                                  .RemainingArgs()
                                  .RemainingRets()
                                  .Attr<int64_t>("kernel_id")
                                  .Attr<int32_t>("n_zero")
                                  .Attr<int32_t>("flags")
                                  .Attr<int64_t>("n_items")
                                  .Attr<int64_t>("stack")
                                  .Attr<int32_t>("n_dev"));
