# Memory Management

NCCL-backed device memory allocation; see [Memory Allocator](../usage/bufferreg.md#mem-allocator) for
usage details. For zero-copy registration of existing buffers, see
[`Communicator.register_buffer()`](communicator/registration.md#nccl.core.Communicator.register_buffer) and
[`Communicator.register_window()`](communicator/registration.md#nccl.core.Communicator.register_window).

## mem_alloc

### nccl.core.mem_alloc(size: int, device: [Device](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Device.html#cuda.core.Device) | int | None = None) → [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer)

Allocates GPU buffer memory using NCCL’s memory allocator.

The actual allocated size may be larger than requested due to buffer
granularity requirements from NCCL optimizations. The returned buffer can
be explicitly freed with [`mem_free()`](#nccl.core.mem_free) or automatically freed when
garbage collected.

* **Parameters:**
  * **size** – Number of bytes to allocate.
  * **device** – Target CUDA device. Defaults to the current device.
* **Returns:**
  A CUDA buffer object backed by NCCL-managed memory. The buffer is
  allocated on the specified device; the current device is restored
  after allocation.

## mem_free

### nccl.core.mem_free(buf: [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer)) → None

Frees memory allocated by [`mem_alloc()`](#nccl.core.mem_alloc).

Explicit deallocation is optional. Memory is automatically freed when the
Buffer object is garbage collected.

* **Parameters:**
  **buf** – The buffer to free.
