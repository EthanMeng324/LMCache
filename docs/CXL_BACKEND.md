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
DAX Device (/dev/dax1.0)
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
   - 提供多进程同步（Lamport's Bakery Algorithm）
   - 处理缓存刷新以确保持久化

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
  
  # CXL 缓存大小限制
  max_cxl_size: 32.0  # GB（默认使用 C 库的 32GB）
```

### 配置说明

- **cxl_num_procs**: 多进程场景下的进程总数
- **cxl_rank**: 当前进程的 rank（0 到 num_procs-1）
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
- **`get_blocking(key)`**: 阻塞读取
- **`remove(key, force=True)`**: 删除 key
- **`pin(key)` / `unpin(key)`**: 固定/取消固定 key
- **`close()`**: 清理资源

## 限制和注意事项

### 硬编码配置

C 库中的硬编码值：
- **DAX 设备路径**: `/dev/dax1.0`
- **设备大小**: 32GB (1UL << 35)

如需修改，需要编辑 `cxl_shm.c` 并重新编译。

### 空间管理

1. **不支持空间回收**: 删除对象后空间不能重用
2. **顺序分配**: 数据按顺序写入，适合追加式场景
3. **元数据限制**: 最大对象数约 2M（`CXL_SHM_MAX_OBJS`）

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
