#ifndef PRIMITIV_MEMORY_POOL_H_
#define PRIMITIV_MEMORY_POOL_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "mixins.h"
#include "error.h"
#include "numeric_utils.h"

using std::cerr;
using std::endl;
using std::make_pair;

/**
 * Memory manager on the device specified by allocator/deleter functors.
 */
class MemoryPool : public mixins::Identifiable<MemoryPool> {
  /**
   * Custom deleter class for MemoryPool.
   */
  class Deleter {
    std::uint64_t pool_id_;
  public:
    explicit Deleter(std::uint64_t pool_id) : pool_id_(pool_id) {}

    void operator()(void *ptr) {
      try {
        MemoryPool::get_object(pool_id_).free(ptr);
      } catch (const primitiv::Error&) {
        // Memory pool already has gone and the pointer is already deleted by
        // the memory pool.
      }
    }
  };

  std::function<void *(std::size_t)> allocator_;
  std::function<void(void *)> deleter_;
  std::vector<std::vector<void *>> reserved_;
  std::unordered_map<void *, std::uint32_t> supplied_;

public:
  /**
   * Creates a memory pool.
   * @param allocator Functor to allocate new memories.
   * @param deleter Functor to delete allocated memories.
   */
  explicit MemoryPool(
      std::function<void *(std::size_t)> allocator,
      std::function<void(void *)> deleter);

  ~MemoryPool();

  /**
   * Allocates a memory.
   * @param size Size of the resulting memory.
   * @return Shared pointer of the allocated memory.
   */
  std::shared_ptr<void> allocate(std::size_t size);

private:
  /**
   * Disposes the memory managed by this pool.
   * @param ptr Handle of the memory to be disposed.
   */
  void free(void *ptr);

  /**
   * Releases all reserved memory blocks.
   */
  void release_reserved_blocks();
};


MemoryPool::MemoryPool(
    std::function<void *(std::size_t)> allocator,
    std::function<void(void *)> deleter)
: allocator_(allocator)
, deleter_(deleter)
, reserved_(64)
, supplied_() {}

MemoryPool::~MemoryPool() {
  // NOTE(odashi):
  // Due to GC-based languages, we chouldn't assume that all memories were
  // disposed before arriving this code.
  while (!supplied_.empty()) {
    free(supplied_.begin()->first);
  }
  release_reserved_blocks();
}

std::shared_ptr<void> MemoryPool::allocate(std::size_t size) {
  static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t), "");

  if (size == 0) return std::shared_ptr<void>();

  static const std::uint64_t MAX_SHIFTS = 63;
  const std::uint64_t shift = numeric_utils::calculate_shifts(size);
  if (shift > MAX_SHIFTS) PRIMITIV_THROW_ERROR("Invalid memory size: " << size);

  void *ptr;
  if (reserved_[shift].empty()) {
    // Allocates a new block.
    try {
      ptr = allocator_(1ull << shift);
    } catch (...) {
      // Maybe out-of-memory.
      // Release other blocks and try allocation again.
      release_reserved_blocks();
      // Below allocation may throw an error when the memory allocation
      // process finally failed.
      ptr = allocator_(1ull << shift);
    }
    supplied_.emplace(ptr, shift);
  } else {
    // Returns an existing block.
    ptr = reserved_[shift].back();
    reserved_[shift].pop_back();
    supplied_.emplace(ptr, shift);
  }

  return std::shared_ptr<void>(ptr, Deleter(id()));
}

void MemoryPool::free(void *ptr) {
  auto it = supplied_.find(ptr);
  if (it == supplied_.end()) {
    PRIMITIV_THROW_ERROR("Detected to dispose unknown handle: " << ptr);
  }
  reserved_[it->second].emplace_back(ptr);
  supplied_.erase(it);
}

void MemoryPool::release_reserved_blocks() {
  for (auto &ptrs : reserved_) {
    while (!ptrs.empty()) {
      deleter_(ptrs.back());
      ptrs.pop_back();
    }
  }
}

#endif  // PRIMITIV_MEMORY_POOL_H_
