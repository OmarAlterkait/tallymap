// tallymap kernel backend, CPU target: ONE generic XLA FFI handler that runs host-compiled kernels.
//
// Python compiles each program's generated C into a shared object (host.py) that exports
//   extern "C" void sj_host_run(int phase, void** p, int np, long long n_items, int tid, int nthreads)
// (phase 0 = sj_prep, 1 = sj_main, 2 = sj_post) and registers its entry point here under the same 64-bit id the
// cuda target uses (tallymap_host_register).  At execution the handler zeroes the first n_zero result buffers and
// runs the phases on T threads, each thread playing one CUDA thread of a one-thread block (blockIdx.x = tid,
// gridDim.x = T): sj_main claims items from the shared counter exactly as on a GPU.
//
// Threads are created here (pthreads, not OpenMP) so that each gets a stack large enough for the item's frame
// (HOST_STACK: per-item local arrays live on the stack, the CPU analogue of CUDA's local memory) and so that the
// kernels need no OpenMP runtime.  T = $OMP_NUM_THREADS when set, else the CPUs this process may run on
// (sched_getaffinity: the cpuset / Slurm allocation), capped by n_items unless flags & 4 (spawn: children).
// The kernels themselves include no XLA headers: only this file does, and it is built once per toolchain.
#include <fenv.h>
#include <pthread.h>
#include <sched.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

typedef void (*SjHostRun)(int, void**, int, long long, int, int);

namespace {

// reserved per thread (pages are committed only when touched): 32 x CUDA's 512 KiB per-thread local-memory limit;
// a kernel whose -fstack-usage bound is over half of it is refused before it runs (host.py, HOST_STACK there)
constexpr size_t HOST_STACK = size_t(16) << 20;

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

struct Job {
  SjHostRun run;
  void** p;
  int np;
  long long n_items;
  int flags;
  int nthreads;
  pthread_barrier_t* bar;
  // the start gate: no thread runs a phase until the whole team exists (a partial team would wait forever at the
  // barrier or in the spawn queue), and a failed pthread_create releases the started threads with abort set
  std::mutex mu;
  std::condition_variable cv;
  int state = 0;                 // 0 waiting, 1 go, 2 abort
};

struct Arg {
  const Job* job;
  int tid;
};

void* worker(void* a) {
  const Arg* w = static_cast<const Arg*>(a);
  Job& j = *const_cast<Job*>(w->job);
  // IEEE: a new thread inherits the creating thread's floating-point environment, and XLA:CPU's threads flush
  // subnormals to zero (measured: 1e-45f + 1e-45f gave 0 in the kernel, as in XLA:CPU; IEEE gives 3e-45)
  fesetenv(FE_DFL_ENV);
  {
    std::unique_lock<std::mutex> lk(j.mu);
    j.cv.wait(lk, [&] { return j.state != 0; });
    if (j.state == 2) return nullptr;
  }
  if (j.flags & 1) {
    j.run(0, j.p, j.np, j.n_items, w->tid, j.nthreads);
    pthread_barrier_wait(j.bar);
  }
  j.run(1, j.p, j.np, j.n_items, w->tid, j.nthreads);
  if (j.flags & 2) {
    pthread_barrier_wait(j.bar);
    j.run(2, j.p, j.np, j.n_items, w->tid, j.nthreads);
  }
  return nullptr;
}

}  // namespace

extern "C" int tallymap_host_register(int64_t id, void* fn) {
  std::lock_guard<std::mutex> lk(g_mu);
  registry()[id] = reinterpret_cast<SjHostRun>(fn);
  return 1;
}

extern "C" int tallymap_host_threads() { return default_threads(); }

static ffi::Error HostLaunchImpl(ffi::RemainingArgs args, ffi::RemainingRets rets, int64_t kernel_id, int32_t n_zero,
                                 int32_t flags, int64_t n_items) {
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
  long long nt = default_threads();
  if (!(flags & 4) && nt > n_items) nt = n_items;
  pthread_barrier_t bar;
  const bool need_bar = (flags & 3) != 0;
  if (need_bar) pthread_barrier_init(&bar, nullptr, (unsigned)nt);
  Job job;
  job.run = run; job.p = p.data(); job.np = (int)p.size(); job.n_items = n_items; job.flags = flags;
  job.nthreads = (int)nt; job.bar = &bar;
  std::vector<Arg> wa(nt);
  std::vector<pthread_t> th(nt);
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, HOST_STACK);
  long long started = 0;
  std::string err;
  // every team member is a new thread with HOST_STACK: the calling thread is XLA's, with a stack tallymap does not
  // control, so it only waits
  for (long long t = 0; t < nt; ++t) {
    wa[t] = Arg{&job, (int)t};
    if (pthread_create(&th[t], &attr, worker, &wa[t]) != 0) { err = "pthread_create failed"; break; }
    ++started;
  }
  pthread_attr_destroy(&attr);
  {
    std::lock_guard<std::mutex> lk(job.mu);
    job.state = err.empty() ? 1 : 2;
  }
  job.cv.notify_all();
  if (!err.empty()) {
    for (long long t = 0; t < started; ++t) pthread_join(th[t], nullptr);
    if (need_bar) pthread_barrier_destroy(&bar);
    return ffi::Error(ffi::ErrorCode::kResourceExhausted, "tallymap: could not start the host kernel's threads");
  }
  for (long long t = 0; t < nt; ++t) pthread_join(th[t], nullptr);
  if (need_bar) pthread_barrier_destroy(&bar);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(TallymapHostLaunch, HostLaunchImpl,
                              ffi::Ffi::Bind()
                                  .RemainingArgs()
                                  .RemainingRets()
                                  .Attr<int64_t>("kernel_id")
                                  .Attr<int32_t>("n_zero")
                                  .Attr<int32_t>("flags")
                                  .Attr<int64_t>("n_items"));
