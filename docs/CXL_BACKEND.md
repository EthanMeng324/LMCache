# CXL Backend 使用文档

## 概述

CXL Backend 使用 `cxl_shm.c` C 库实现 CXL (Compute Express Link) 存储后端。它保持了 Python `StorageBackendInterface` 的接口，但底层使用 C 实现进行内存管理，提供高性能的持久化存储。

## 设计原理

### 架构

```
Python Layer (cxl_backend.py)
    ↓
Python Binding (cxl_shm_binding.py)
    ↓
C Library (cxl_shm.c)
    ↓
DAX Device (e.g. /dev/dax0.0)
```

### 核心组件

1. **CXL Backend (`cxl_backend.py`)**
   - 实现 `StorageBackendInterface`
   - 管理 key 到 CXL 对象的映射
   - 处理缓存策略和空间管理

2. **C Library Binding (`cxl_shm_binding.py`)**
   - 使用 `ctypes` 调用 C 函数
   - 定义 C 结构体的 Python 等价物
   - 提供 Python 友好的接口

3. **C Library (`cxl_shm.c`)**
   - 管理 DAX 设备的内存映射
   - 实现顺序分配存储
   - 提供多进程同步（“smart lock”，Lamport/Bakery 风格）
   - 处理缓存刷新以确保持久化

4. **CXL Worker (`CxlWorker` in `cxl_backend.py`)**
   - 一个后端内部的 **异步任务调度器**（单独线程 + 线程池）
   - 主要服务于 `submit_put_task()` / `submit_prefetch_task()` 这类“后台搬运数据”的操作
   - 提供 **优先级队列**、**任务去重**、以及 **prefetch Future 的追踪/等待**

### CXL worker 的意义是什么？

`CxlWorker` 不是 “CXL 设备的 worker”，而是 **CxlBackend 的后台执行器**，解决的是 Python 侧的工程问题：

- **避免阻塞调用线程**：`submit_put_task()`（写入）与 `submit_prefetch_task()`（预取）都可能触发较大的内存拷贝和 `clflush`，放到后台线程执行可以让前台逻辑更轻量。
- **统一串行化/资源保护**：内部 `ThreadPoolExecutor(max_workers=1)` 让这些重操作在 worker 里默认串行执行，减少并发写/读导致的抖动（C 侧也有跨进程锁，但 Python 侧的串行化能降低竞争面）。
- **优先级控制**：`prefetch` 优先级最高（priority=0），其次是 `delete`（priority=1），最后是 `put`（priority=2）。这背后的意图是：**读路径相关（prefetch）优先于写入**，尽量降低命中时的尾延迟。
- **任务去重与“在途状态”**：
  - `put_tasks`：防止同一个 key 重复提交写入（`exists_in_put_tasks()`）。
  - `prefetch_tasks`：记录 key 的 prefetch 是否“已排队/运行中/已拿到 Future”，并允许 `get_blocking()` 在需要时 `wait_prefetch_task()` 等待结果。

> 备注：当前代码里 `delete` 类型任务的通道已预留，但 `remove()` 走的是同步路径；如果未来需要把删除也放后台，这个机制已经具备基本形态。

### 存储机制

#### 顺序分配

CXL Backend 使用顺序分配策略：
- 数据从 `curr_offset` 开始顺序写入
- 不支持空间回收（删除后空间不能重用）
- 适合写入一次、多次读取的场景

#### 元数据管理

- 使用哈希表（`mem_hash_t`）管理对象元数据
- 每个对象包含：名称、偏移量、大小、使用标志
- 元数据区域大小：约 67MB（2M 对象 × 32 字节）

#### Key 处理

- 对象名称限制：20 字节（`CXL_SHM_ONAME_LEN`）
- 如果 key 超过 20 字节，自动使用 SHA256 哈希
- 哈希格式：`H{sha256[:16]}` (17 字节)

## 编译和安装

### ⚠️ 重要：必须先编译 C 库

在使用 CXL Backend **之前**，必须先编译 C 库。这是**必需步骤**，无论是：
- 运行测试
- 运行整个 lmcache
- 使用 CXL Backend 功能

都需要先完成编译。

### 编译步骤

```bash
# 1. 编译 C 库（使用 Makefile，推荐）
make -f Makefile.cxl_shm

# 2. 将库文件复制到项目目录
cp libcxl_shm.so lmcache/v1/storage_backend/
```

### 手动编译（如果 Makefile 不可用）

```bash
gcc -shared -fPIC -O2 -std=c11 -fvisibility=hidden \
    -o libcxl_shm.so cxl_shm.c mem_hash.c \
    -lrt -lpthread -march=native

cp libcxl_shm.so lmcache/v1/storage_backend/
```

### 库文件查找顺序

Python 绑定会按以下顺序查找库文件：
1. `lmcache/v1/storage_backend/libcxl_shm.so` ⭐ **推荐位置**
2. `lmcache/libcxl_shm.so`
3. 当前目录的 `libcxl_shm.so`
4. 系统路径中的 `libcxl_shm.so`（需要设置 `LD_LIBRARY_PATH`）

### 验证编译

```bash
# 检查库文件是否存在
ls -lh libcxl_shm.so

# 检查库依赖
ldd libcxl_shm.so

# 检查导出符号
nm -D libcxl_shm.so | grep cxl_shm
```

### 故障排查

如果遇到 "Failed to load libcxl_shm.so" 错误：
1. 确认已执行 `make -f Makefile.cxl_shm`
2. 确认 `libcxl_shm.so` 已复制到 `lmcache/v1/storage_backend/`
3. 检查文件权限：`chmod +x lmcache/v1/storage_backend/libcxl_shm.so`

## 配置

在 `LMCacheEngineConfig` 的 `extra_config` 中配置：

```yaml
extra_config:
  # CXL 进程配置（用于多进程场景）
  cxl_num_procs: 1  # 进程数量
  cxl_rank: 0       # 当前进程 rank (0-based)
  
  # DAX 设备路径（可选）
  # 优先级：extra_config.cxl_dax_device > 环境变量 LMCACHE_CXL_DAX_DEVICE > C 侧默认值
  cxl_dax_device: /dev/dax1.0
  
  # CXL 缓存大小限制
  max_cxl_size: 32.0  # GB（默认使用 C 库的 32GB）
```

### 配置说明

- **cxl_num_procs**: 多进程场景下的进程总数
- **cxl_rank**: 当前进程的 rank（0 到 num_procs-1）
- **cxl_dax_device**: DAX 设备路径（最终会设置到环境变量 `LMCACHE_CXL_DAX_DEVICE`，供 C 库 `cxl_shm_init()` 使用）
- **max_cxl_size**: 最大缓存大小（GB），用于 Python 层的空间管理

## 使用方式

### 基本使用

```python
from lmcache.v1.storage_backend.cxl_backend import CxlBackend
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
import asyncio

# 创建配置
config = LMCacheEngineConfig(
    cache_policy="LRU",
    local_cpu=True,
    extra_config={
        "cxl_num_procs": 1,
        "cxl_rank": 0,
        "max_cxl_size": 32.0,
    }
)

# 创建 local CPU backend
local_cpu_backend = LocalCPUBackend(...)

# 创建 CXL backend
loop = asyncio.get_event_loop()
backend = CxlBackend(
    config=config,
    loop=loop,
    local_cpu_backend=local_cpu_backend,
    dst_device="cuda",
)

# 使用方式与原来的 CxlBackend 完全相同
from lmcache.utils import CacheEngineKey
key = CacheEngineKey(...)

# 写入
backend.submit_put_task(key, memory_obj)

# 读取
memory_obj = backend.get_blocking(key)

# 删除
backend.remove(key)

# 清理
backend.close()
```

### API 接口

CXL Backend 实现了 `StorageBackendInterface`，主要方法：

- **`contains(key, pin=False)`**: 检查 key 是否存在
- **`submit_put_task(key, memory_obj)`**: 异步写入
- **`submit_prefetch_task(key)`**: 异步预取（需要启用 `local_cpu`）
- **`get_blocking(key)`**: 阻塞读取
- **`remove(key, force=True)`**: 删除 key
- **`pin(key)` / `unpin(key)`**: 固定/取消固定 key
- **`close()`**: 清理资源

## 对外 API 的完整流程（从 Python 到 C/DAX）

这一节按“你调用的 API → Python 层做什么 → 绑定层调用什么 → C 库内部发生什么”把链路串起来。

### 1) `contains(key, pin=False)`

- **Python 层入口**：`CxlBackend.contains()`
- **Python 做的事**：
  - 先查 `self.dict`（缓存策略维护的 key→元数据映射）。
  - 如果不在 `dict`，会尝试“去 CXL 里探测是否存在”：
    - 计算 C 侧对象名：`_get_cxl_key()`（若 `key.to_string()` 超过 20 字节，改用 `H{sha256[:16]}`）。
    - 调用绑定层 `open_obj()`。
  - 若 `open_obj()` 成功：把一个 `DiskCacheMetadata(path, size, shape=None, dtype=None, fmt=None)` 填回 `dict`，并把 `CxlShmHnd` 放进 `key_handles`。
  - 如果 `pin=True`：会对 `dict[key]` 做 pin，并把 key 记录到 `keys_in_request`，供后续 `touch_cache()` 统一更新“命中顺序”。
- **绑定层调用**：`CxlShmWrapper.open_obj(name)` → `_lib.cxl_shm_open_obj(name, &hnd)`
- **C 侧发生什么**（`cxl_shm_open_obj`）：
  - 进入跨进程 smart lock 临界区。
  - 在 `meta->objs[]`（对象元数据数组，哈希定位）里找到 `name` 对应条目。
  - 返回 `hnd->obj`（元数据指针）与 `hnd->mapped_addr = dax_addr + offset`。
  - 对映射区域做 `clflush_region_with_mfence`（确保可见性/持久化一致性）。

### 2) `submit_put_task(key, memory_obj)`

- **Python 层入口**：`CxlBackend.submit_put_task()`
- **Python 做的事（前台阶段）**：
  - 用 `CxlWorker.put_tasks` 做 **重复提交去重**（同 key 正在写就直接跳过）。
  - 以 `required_size = memory_obj.get_physical_size()` 做 **Python 侧空间压力判断**：
    - 如果 `current_cache_size + required_size > max_cache_size`，按 cache policy 找候选并 `batched_remove(..., force=False)` 做淘汰（注意：这只是 Python 侧“容量账本”，C 库是顺序分配，删除不会回收 offset）。
  - `memory_obj.ref_count_up()` 确保后台写完之前对象不被释放。
  - 把真正写入动作交给 `CxlWorker`：`submit_task("put", async_save_bytes_to_cxl, key=..., memory_obj=...)`
- **后台阶段（worker 线程池）**：`async_save_bytes_to_cxl()`
  - 取 `memory_obj.byte_array`（KV chunk 序列化后的字节视图）。
  - 调 `write_cxl_mmap(key, buffer, memory_obj)`：
    - 绑定层：`CxlShmWrapper.create(name, size)` → `_lib.cxl_shm_create(...)`
    - Python 用 `ctypes.memmove(mapped_addr, buffer_bytes, size)` 把数据写进 DAX 映射地址
    - 调 `flush(mapped_addr, size)` → `_lib.clflush_region_with_mfence(...)` 确保持久化
  - 写入成功后更新 Python 元数据：`insert_key(key, memory_obj)`（把 shape/dtype/fmt 等记录进 `dict`）
  - `memory_obj.ref_count_down()` + 从 `put_tasks` 移除 key
- **C 侧发生什么**（`cxl_shm_create`）：
  - smart lock 临界区内检查是否已存在同名对象
  - 找一个空的元数据槽位（哈希定位，要求 `in_use==0`）
  - 用 `meta->head.curr_offset` **顺序分配**一段空间（会 cacheline 对齐）
  - 填 `obj->offset/size/in_use/name`，并返回 `mapped_addr = dax_addr + offset`
  - 更新 `curr_offset = curr_offset + size`（只增不减）

### 3) `submit_prefetch_task(key)`（需要 `local_cpu=True`）

- **目的**：把 CXL/DAX 上的数据 **提前搬到 local CPU backend**，让随后的 `get_blocking()` 走更快的“本地命中路径”。
- **Python 层入口**：`CxlBackend.submit_prefetch_task()`
- **Python 做的事**：
  - 确认 key 在 `dict`（否则无法知道 shape/dtype/fmt 并分配目标 buffer）。
  - 对 key 做 cache policy 的命中更新，并检查 `prefetch_tasks` 去重（同 key 只允许一个 prefetch in-flight）。
  - 通过 `local_cpu_backend.allocate(shape, dtype, fmt)` 分配一个目标 `MemoryObj`（用于接收字节）。
  - 临时 pin 住 `dict[key]`，避免被淘汰。
  - 交给 `CxlWorker`：`submit_task("prefetch", async_load_bytes_from_cxl, key=..., memory_obj=...)`
- **后台阶段**：`async_load_bytes_from_cxl(key, memory_obj)`
  - 取 `memory_obj.byte_array` 作为目标 buffer
  - 调 `read_cxl_mmap(key, buffer)`：绑定层 `open_obj()` 拿到 `mapped_addr`，再从映射区拷贝到 buffer
  - 解除 pin：`dict[key].unpin()`
  - 把 `memory_obj` 交给 `local_cpu_backend.submit_put_task(key, memory_obj)`，完成“本地落盘/登记”
  - 从 `prefetch_tasks` 移除 key，并返回该 `memory_obj`（供 `wait_prefetch_task()` 拿结果）

### 4) `get_blocking(key)`

- **Python 层入口**：`CxlBackend.get_blocking()`
- **完整路径分两类**：

**A. 先等 prefetch（如果该 key 正在 prefetch）**
- `wait_prefetch_task(key)`：
  - 若 key 不在 `prefetch_tasks`：表示没有 prefetch，直接走 B 路径
  - 若 value 是 `Future`：阻塞 `future.result()` 拿到 `MemoryObj`
- 拿到 `MemoryObj` 后，会额外检查 `local_cpu_backend.contains(key, pin=True)`，如果本地已登记则直接返回这个 `memory_obj`

**B. 直接从 CXL/DAX 读取**
- 先确保 key 存在：
  - key 不在 `dict` 时，调用 `contains(key, pin=False)` 走“探测/打开对象”的逻辑；不存在就返回 `None`
- 读取路径：
  - 从 `dict[key]` 取出 `shape/dtype/fmt`（这些一般在 put 时由 `insert_key()` 写入）
  - 调 `load_bytes_from_cxl(key, dtype, shape, fmt)`：
    - 绑定层 `open_obj()` → 拿到 `mapped_addr` 与 `size`
    - Python 用 `ctypes` 把映射地址包成 `torch.frombuffer(...)`，再 `view(dtype).view(shape)` 得到一个“直接指向 CXL 映射区”的 Tensor（零拷贝语义）
    - 保存 `hnd` 到 `key_handles`，便于后续 `close()`/`remove()`

> 注意：如果 key 是通过 `contains()` 被“回填”进 `dict` 的，它的 `shape/dtype/fmt` 可能是 `None`（因为 C 侧元数据里只有 size/offset/name），此时 `get_blocking()` 需要调用方保证元数据来源正确（通常由 put 路径写入）。

### 5) `remove(key, force=True)`

- **Python 层入口**：`CxlBackend.remove()`
- **Python 做的事**：
  - 从 `dict` 移除元数据并更新 usage/current_cache_size
  - 如果 `key_handles` 里有 handle：
    - 调 `cxl_shm.destroy(hnd)` → `_lib.cxl_shm_destroy_from_hnd(&hnd)`
    - 再调 `cxl_shm.close(hnd)` → `_lib.cxl_shm_close(&hnd)`
  - 根据 `force` 更新 cache policy，并可选向 `lmcache_worker` 发 `KVEvictMsg`
- **C 侧发生什么**（`cxl_shm_destroy_from_hnd`）：
  - smart lock 临界区内：
    - `memset(mapped_addr, 0, size)` 并 `clflush`，把对象数据清零
    - 清空对象元数据（`in_use=0`，`memset(obj_meta,0,...)` 并 flush）
  - **不会回收 `curr_offset`**：因此 C 侧空间仍然是“只增不减”的顺序分配模型

### 6) `pin(key)` / `unpin(key)` / `touch_cache()`

- **pin/unpin**：纯 Python 侧元数据操作（`DiskCacheMetadata.pin()/unpin()`），用于配合淘汰策略避免热点 key 被驱逐。
- **touch_cache()**：把 `contains(..., pin=True)` 期间累积的 `keys_in_request` 倒序回放到 cache policy，用于维护 prefix/suffix 命中顺序（实现细节依赖具体 cache policy）。

### 7) `close()`

- **Python 层**：
  - 遍历 `key_handles` 调 `_lib.cxl_shm_close(&hnd)` 释放映射 handle（当前 C 实现 close 基本只是把指针清空）
  - 调 `_lib.cxl_shm_finalize()`（rank==0 会把 DAX 映射区域清零并把 `initialized` 置 0）
  - 关闭 `CxlWorker`（停止后台线程、shutdown executor）

## 限制和注意事项

### 硬编码配置

C 库中的硬编码值：
- **设备大小**: 32GB (`1UL << 35`)

如需修改，需要编辑 `cxl_shm.c` 并重新编译。

> DAX 设备路径在 C 库侧默认是 `/dev/dax0.0`，但实际使用时推荐通过
> - `extra_config.cxl_dax_device` 或
> - 环境变量 `LMCACHE_CXL_DAX_DEVICE`
> 来覆盖（Python 初始化时会写入该环境变量）。

### 空间管理

1. **不支持空间回收**: 删除对象后空间不能重用
2. **顺序分配**: 数据按顺序写入，适合追加式场景
3. **元数据限制**: 最大对象数约 2M（`CXL_SHM_MAX_OBJS`）

### 常见误区 / 注意事项

1. **`max_cxl_size` 主要是 Python 层的“容量账本”**  
   Python 会用它来触发淘汰（更新 `current_cache_size` 并调用 `remove`），但 C 库的分配策略是顺序分配（`curr_offset` 只增不减），所以 **删除不会让 C 侧 offset 回退**。这意味着长时间运行且写入模式偏“不断产生新 key”时，C 侧可能仍会最终写满 32GB。

2. **C 侧元数据不包含 shape/dtype/fmt**  
   `cxl_shm.c` 的对象元数据只有 `name/offset/size/in_use`。因此如果你只靠 `contains()` 去“探测”一个历史 key，然后直接 `get_blocking()`，可能拿不到 `shape/dtype/fmt`（因为 Python 侧回填的是 `None`）。正常路径应该由 put 写入时 `insert_key()` 把这些信息写进 `dict`（或由更高层的元数据系统补齐）。

### Key 长度

- 对象名称最大 20 字节
- 超过时自动使用哈希（冲突概率极低：1/2^64）
- 哈希映射会缓存，确保一致性

### 多进程支持

- 支持多进程访问同一 DAX 设备
- 使用 Lamport's Bakery Algorithm 进行进程间同步
- 需要正确设置 `num_procs` 和 `rank`

### 持久化

- 数据写入后自动 flush 到 DAX 设备
- 使用 `clflush_region_with_mfence` 确保持久化
- 重启后数据仍然存在（如果设备支持）

## 故障排查

### 库加载失败

```
RuntimeError: Failed to load libcxl_shm.so
```

**解决方案**：
1. 确保已编译 `libcxl_shm.so`
2. 检查库文件路径和权限
3. 使用 `LD_LIBRARY_PATH` 设置库路径

### 初始化失败

```
RuntimeError: Failed to initialize CXL shared memory: -1
```

**可能原因**：
1. DAX 设备不存在或不可访问
2. 权限不足
3. 设备已被其他进程使用

**解决方案**：
1. 检查 `/dev/dax1.0` 是否存在：`ls -la /dev/dax1.0`
2. 检查权限：`sudo chmod 666 /dev/dax1.0`
3. 检查占用：`lsof /dev/dax1.0`

### 对象创建失败

**可能原因**：
1. 空间不足
2. 元数据数组已满
3. 哈希冲突（极罕见）

**解决方案**：
1. 检查设备空间使用情况
2. 减少对象数量
3. 重置设备（如果支持）

## 性能考虑

### 优势

1. **C 实现**: 无 Python GIL 限制，直接内存操作
2. **持久化**: 数据直接写入 DAX 设备，无需额外拷贝
3. **多进程**: 支持多进程并发访问

### 潜在瓶颈

1. **ctypes 调用开销**: 通常可忽略
2. **顺序分配**: 空间浪费，不支持回收
3. **多进程锁竞争**: 高并发时可能成为瓶颈

## 测试

运行单元测试：

```bash
pytest tests/v1/storage_backend/test_cxl_backend.py -v
```

测试覆盖：
- Key 长度处理（短 key 和长 key）
- 基本操作（初始化、contains、pin/unpin、remove）
- 内存操作（read/write）
- 清理操作（close）

## 与原始实现的对比

| 特性 | 原始实现 | C 库实现 |
|------|---------|---------|
| 实现语言 | Python | C (cxl_shm.c) |
| 存储方式 | Slot-based (8MB) | Sequential allocation |
| 大小限制 | 8MB per slot | 无固定限制 |
| 空间回收 | 支持（覆盖 slot） | 不支持 |
| 多进程 | 不支持 | 支持 |
| 性能 | Python 开销 | C 性能 |
| 持久化 | 需要手动 flush | 自动 flush |

## 参考

- `cxl_shm.c`: C 库实现
- `cxl_shm.h`: C 库头文件
- `cxl_shm_binding.py`: Python 绑定
- `cxl_backend.py`: Python backend 实现
