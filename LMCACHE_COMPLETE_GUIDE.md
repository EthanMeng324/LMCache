# LMCache完整技术指南

本文档整合了LMCache的核心技术文档，包括请求流程、Lookup机制、Storage Backend系统和KV移动能力。

---

## 目录

1. [整体架构和请求流程](#1-整体架构和请求流程)
2. [Lookup机制详解](#2-lookup机制详解)
3. [Storage Backend系统](#3-storage-backend系统)
4. [KV移动能力](#4-kv移动能力)

---

## 1. 整体架构和请求流程

### 1.1 架构概述

LMCache通过实现vLLM的`KVConnectorBase_V1`接口来集成，主要包含两个角色：
- **Scheduler端**：负责缓存查找和调度决策
- **Worker端**：负责KV缓存的加载和保存

### 1.2 初始化阶段

**代码位置**：`lmcache/integration/vllm/lmcache_connector_v1.py`

#### Scheduler端初始化
- 代码：`LMCacheConnectorV1Impl.__init__()` (第656-825行)
- 创建`lookup_client`用于缓存查找
- 可选创建`lmcache_engine`（如果启用`enable_scheduler_bypass_lookup`）
- 初始化`_unfinished_requests`、`load_specs`、`_request_trackers`等数据结构

#### Worker端初始化
- 代码：`LMCacheConnectorV1Impl.__init__()` (第720-756行)
- 调用`_init_lmcache_engine()`创建LMCache引擎 (第481-639行)
- 创建`lookup_server`用于接收查找请求
- 创建`offload_server`用于KV缓存卸载
- 初始化`layerwise_retrievers`和`layerwise_storers`（如果启用layerwise模式）

### 1.3 Scheduler端 - 缓存查找阶段

#### get_num_new_matched_tokens()

**调用时机**：vLLM调度器在分配KV cache blocks之前调用

**代码位置**：`lmcache/integration/vllm/vllm_v1_adapter.py:1472-1589`

**流程**：
1. 检查lookup_client缓存中是否已有结果
2. 如果没有，执行实际查找：
   - 提取token IDs
   - 处理multimodal features（如果有）
   - 调用`lookup_client.lookup()`进行缓存查找
3. 计算需要分配的tokens数量：`need_to_allocate = num_external_hit_tokens - num_computed_tokens`
4. 创建`LoadSpec`并存储到`load_specs`字典中
5. 返回`need_to_allocate`

#### update_state_after_alloc()

**调用时机**：vLLM的KV cache manager分配完blocks后调用

**代码位置**：`lmcache/integration/vllm/vllm_v1_adapter.py:1592-1661`

**流程**：
1. 清除lookup_client中的本地状态
2. 处理disaggregation spec（如果有）
3. 将request添加到`_unfinished_requests`
4. 验证tokens数量并设置`can_load`标志

#### build_connector_meta()

**调用时机**：vLLM调度器在构建scheduler output时调用

**代码位置**：`lmcache/integration/vllm/vllm_v1_adapter.py:1664-1854`

**流程**：
1. 清理已完成的requests
2. 处理新请求：创建`RequestTracker`和`ReqMeta`
3. 处理已缓存的请求：更新状态，处理preemption情况
4. 返回`LMCacheConnectorMetadata`

### 1.4 Worker端 - KV缓存加载阶段

#### start_load_kv()

**调用时机**：vLLM在forward pass开始前调用

**代码位置**：`lmcache/integration/vllm/vllm_v1_adapter.py:940-1066`

**流程**：
1. 获取connector metadata
2. 遍历metadata中的requests，对每个需要加载的request：
   - 计算token_mask（标记哪些tokens需要从LMCache加载）
   - 如果启用layerwise模式：创建layerwise retriever
   - 如果非layerwise模式：调用`lmcache_engine.retrieve()`进行批量加载

#### wait_for_layer_load()

**调用时机**：vLLM在每个attention layer的forward pass中调用（仅layerwise模式）

**代码位置**：`lmcache/integration/vllm/vllm_v1_adapter.py:1142-1163`

**流程**：推进layerwise_retrievers到下一层，实现pipelining

### 1.5 Worker端 - KV缓存保存阶段

#### save_kv_layer()

**调用时机**：vLLM在每个attention layer的forward pass后调用（仅layerwise模式）

**代码位置**：`lmcache/integration/vllm/vllm_v1_adapter.py:1166-1266`

**流程**：
1. 第一层调用时创建所有layerwise_storers
2. 后续层调用时推进所有storers
3. 实现pipelining：在计算下一层时，异步保存当前层的KV cache

#### wait_for_save()

**调用时机**：vLLM在forward pass结束后调用

**代码位置**：`lmcache/integration/vllm/vllm_v1_adapter.py:1269-1363`

**流程**：
1. 确保所有异步保存操作完成
2. 更新已保存的tokens数量
3. 释放pinned的KV cache

---

## 2. Lookup机制详解

### 2.1 Lookup Client架构

**Lookup Client**是LMCache中负责在**Scheduler端**查询KV缓存的核心组件。

```
┌─────────────────────────────────────────────────────────┐
│                    vLLM Scheduler                        │
│  ┌──────────────────────────────────────────────────┐   │
│  │  LookupClient (Scheduler端)                      │   │
│  │  - 接收request的token IDs                        │   │
│  │  - 查询LMCache中是否有缓存                       │   │
│  │  - 返回命中tokens数量                             │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                        │
                        │ ZMQ/IPC通信
                        ▼
┌─────────────────────────────────────────────────────────┐
│                    vLLM Worker                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  LookupServer (Worker端)                         │   │
│  │  - 接收lookup请求                                 │   │
│  │  - 调用LMCacheEngine.lookup()                    │   │
│  │  - 返回命中tokens数量                             │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Lookup Client类型

#### LMCacheLookupClient（同步模式）

**代码位置**：`lmcache/v1/lookup_client/lmcache_lookup_client.py`

**特点**：
- 使用ZMQ进行同步通信
- 阻塞式查找，立即返回结果
- 支持多rank（TP/PP）并行查找

**工作流程**：
1. 将token IDs转换为chunk hashes
2. 通过ZMQ socket发送查找请求到所有ranks
3. 等待所有ranks返回结果
4. 取最小值作为最终结果（确保所有rank都能加载到相同的KV cache）

#### LMCacheAsyncLookupClient（异步模式）

**代码位置**：`lmcache/v1/lookup_client/lmcache_async_lookup_client.py`

**特点**：
- 使用ZMQ进行异步通信
- 非阻塞式查找，立即返回None
- 后台线程处理响应
- 支持prefetch（预取）

#### LMCacheBypassLookupClient（直连模式）

**代码位置**：`lmcache/v1/lookup_client/lmcache_lookup_client_bypass.py`

**特点**：
- 直接调用LMCacheEngine，不通过ZMQ通信
- 适用于MLA（Multi-Level Attention）场景
- 只在rank 0执行查找

### 2.3 完整Lookup流程

```
get_num_new_matched_tokens()
  └─> LookupClient.lookup()
      └─> TokenDatabase.process_tokens()  [生成hashes]
      └─> ZMQ发送到所有ranks
          └─> LookupServer接收
              └─> LMCacheEngine.lookup()
                  └─> TokenDatabase.process_tokens()  [生成CacheEngineKey]
                  └─> StorageManager.batched_contains()
                      └─> 按优先级遍历backends
                          ├─> LocalCPUBackend.batched_contains()
                          ├─> LocalDiskBackend.batched_contains()
                          └─> RemoteBackend.batched_contains()
                  └─> 返回命中的tokens数量
          └─> 收集所有ranks的响应
          └─> 取最小值
          └─> 返回need_to_allocate
```

### 2.4 TokenDatabase - 生成Cache Keys

**代码位置**：`lmcache/v1/token_database.py`

**功能**：将token序列转换为chunk keys

**ChunkedTokenDatabase实现**：
- 将tokens按chunk_size（默认256）分块
- 对每个chunk计算prefix hash
- 生成`CacheEngineKey`对象

### 2.5 StorageManager - 批量查询存储后端

**代码位置**：`lmcache/v1/storage_backend/storage_manager.py:794-832`

**流程**：
1. 遍历所有active backends（按优先级顺序）
2. 调用backend的`batched_contains()`
3. 如果所有keys都命中了，提前退出
4. 如果部分keys命中，继续查询剩余的keys

**关键设计点**：
- **Prefix Matching**：只返回连续前缀的命中数量，遇到第一个miss就停止
- **多Backend优先级查询**：按优先级顺序查询，找到就提前退出
- **Pin机制**：在lookup阶段pin住KV cache，防止在加载前被evict

### 2.6 各Backend的查找实现

| 存储后端 | 代码位置 | 查找方式 | 时间复杂度 |
|---------|---------|---------|-----------|
| **LocalCPUBackend** | `local_cpu_backend.py:119` | dict查找 | O(1) |
| **LocalDiskBackend** | `local_disk_backend.py:180` | dict查找文件路径 | O(1) |
| **RemoteBackend** | `remote_backend.py:132` | 网络查询 | O(网络延迟) |

---

## 3. Storage Backend系统

### 3.1 Backend优先级顺序

**核心原理**：Backend的优先级顺序由`CreateStorageBackends()`函数中的创建顺序决定，因为使用的是`OrderedDict`，插入顺序就是遍历顺序。

**代码位置**：`lmcache/v1/storage_backend/__init__.py:106-230`

#### 创建顺序（优先级从高到低）

1. **PDBackend**（如果`enable_pd=True`）
2. **LocalCPUBackend**（如果`max_local_cpu_size > 0`）
3. **P2PBackend**（如果`enable_p2p=True`）
4. **NixlStorageBackend**（如果`enable_nixl_storage=True`）
5. **LocalDiskBackend**（如果`local_disk=True`且`max_local_disk_size > 0`）
6. **GdsBackend**（如果`gds_path`不为None）
7. **RemoteBackend**（如果`remote_url`不为None）
8. **Storage Plugins**（动态加载的自定义backend）

**关键点**：
- 优先级不是硬编码的，而是由创建顺序决定
- 只有满足配置条件的backend才会被创建
- 可以通过storage plugins添加自定义backend

### 3.2 Backend之间的交互机制

#### 核心设计原则

**重要发现**：**Backend之间没有自动的eviction迁移机制**。每个backend独立管理自己的存储空间和eviction策略。

#### 存储时的行为

**代码位置**：`lmcache/v1/storage_backend/storage_manager.py:374-419`

```python
def batched_put(..., location: Optional[str] = None):
    # 如果指定了location，只写入该backend
    # 否则写入所有backend
    for backend_name, backend in self.storage_backends.items():
        if location and backend_name != location:
            continue
        backend.batched_submit_put_task(...)
```

**关键点**：
- **默认行为**：如果没有指定`location`，KV cache会**同时写入所有backend**
- **指定location**：如果指定了`location`，只写入该backend
- **独立存储**：每个backend独立存储，不是层级关系

#### 各Backend的Eviction策略

**LocalCPUBackend**：
- 代码位置：`local_cpu_backend.py:426-498`
- Eviction行为：从`hot_cache`中删除，**不会自动迁移到下一级backend**

**LocalDiskBackend**：
- 代码位置：`local_disk_backend.py:291-344`
- Eviction行为：从磁盘删除文件，**不会自动迁移到RemoteBackend**

#### 读取时的行为

**代码位置**：`lmcache/v1/storage_backend/storage_manager.py:421-445`

```python
def get(self, key: CacheEngineKey, location: Optional[str] = None):
    for backend_name, backend in self.get_active_storage_backends(location):
        memory_obj = backend.get_blocking(key)
        if memory_obj:
            # 关键：如果从非LocalCPUBackend获取到，自动写入LocalCPUBackend作为缓存
            if (
                backend_name not in ["LocalCPUBackend", "PDBackend"]
                and "LocalCPUBackend" in self.storage_backends
            ):
                local_cpu_backend.submit_put_task(key, memory_obj)
            return memory_obj
```

**关键机制**：
- **自动缓存**：从Disk/Remote获取的KV cache会自动写入LocalCPUBackend
- **性能优化**：下次访问时可以从CPU内存快速获取
- **例外**：PDBackend获取的不缓存（因为PD是disaggregation场景）

### 3.3 NixlStorageBackend详解

**NIXL (NVIDIA Index Library)** 是NVIDIA提供的高性能存储和传输库，支持：
- GPU Direct Storage (GDS)
- 高性能文件系统（HF3FS）
- 对象存储（OBJ）
- POSIX文件系统

**代码位置**：`lmcache/v1/storage_backend/nixl_storage_backend.py`

**存储位置**：
- **GDS / GDS_MT**：NVMe SSD（通过GDS路径访问）
- **POSIX**：本地文件系统路径
- **HF3FS**：HF3FS文件系统路径
- **OBJ**：对象存储系统

**特点**：
- **高性能**：利用NIXL的零拷贝和GPU Direct Storage
- **低延迟**：直接GPU到存储的传输
- **大容量**：支持TB级别的存储

### 3.4 PDBackend详解

**PD (Persistent Disaggregation)** 是LMCache用于**prefill-decode disaggregation**场景的backend。

**应用场景**：
- **Prefill服务器**：计算prompt的KV cache
- **Decode服务器**：使用prefill的KV cache进行生成
- **分离部署**：prefill和decode可以在不同的服务器上

**代码位置**：`lmcache/v1/storage_backend/pd_backend.py`

#### 两种角色

**1. Sender (Prefiller)**：
- **行为**：**不存储KV cache**，直接传输到receiver
- 代码注释：`At the sender side, it will never save anything but directly write the data to the receiver side.`

**2. Receiver (Decoder)**：
- **行为**：接收并存储KV cache
- **存储位置**：`self.data: dict[CacheEngineKey, MemoryObj]`（内存）
- **特点**：临时存储，用于decode阶段

**传输机制**：
- 使用ZMQ进行数据传输
- 支持多种传输方式（RDMA、ZMQ等）
- 异步传输，不阻塞prefill计算

---

## 4. KV移动能力

### 4.1 当前实现的能力

#### 1. 跨节点移动（Cross-Node Move）- ✅ 已实现

**功能**：在不同LMCache实例之间移动KV cache

**代码位置**：
- Controller API: `lmcache/v1/cache_controller/controllers/kv_controller.py:107`
- Executor: `lmcache/v1/cache_controller/executor.py:281`
- Worker处理: `lmcache/v1/cache_controller/worker.py:392-423`
- Engine实现: `lmcache/v1/cache_engine.py:1021-1085`

**工作流程**：
```
Controller API
    │
    │ 1. 接收move请求
    │ 2. 找到源和目标worker
    │ 3. 发送MoveWorkerMsg到源worker
    │
    ▼
Source Worker (LMCacheEngine.move())
    │
    │ 4. lookup()查找old_position中的KV cache
    │ 5. batched_get()从old_position获取memory objects
    │ 6. 通过P2PBackend传输到new_position
    │ 7. 如果do_copy=False，删除old_position中的KV
    │
    ▼
Target Worker (P2PBackend接收)
    │
    │ 8. 接收并存储到new_position指定的backend
```

**限制**：
- 需要启用P2PBackend (`enable_p2p=True`)
- 需要NIXL支持（`transfer_channel="nixl"`）
- **目前只支持移动到`LocalCPUBackend`**
- 需要Controller支持（`enable_controller=True`）

#### 2. 读取时自动缓存 - ✅ 已实现

**功能**：从慢速backend读取时，自动缓存到LocalCPUBackend

**代码位置**：`lmcache/v1/storage_backend/storage_manager.py:421-445`

**特点**：
- **自动触发**：从Disk/Remote读取时自动执行
- **性能优化**：下次访问可以从CPU快速获取
- **单向**：只支持从慢速→快速（Disk/Remote → CPU）

#### 3. 指定location存储 - ✅ 已实现

**功能**：通过`location`参数控制KV cache存储到特定backend

**代码位置**：`lmcache/v1/storage_backend/storage_manager.py:374-419`

**使用场景**：
- 只存储到特定backend（如只存Disk，不存CPU）
- 手动控制存储位置

### 4.2 未实现/部分实现的能力

#### 1. 同一节点内不同Backend之间的移动 - ❌ 未实现

**代码位置**：`lmcache/v1/cache_controller/worker.py:399-411`

**状态**：代码框架存在，但抛出`NotImplementedError`

```python
# Intra node move
if new_position[0] == self.lmcache_worker_internal_url:
    # TODO(Jiayi): currently we only support moving from
    # local disk to local cpu.
    assert old_position[1] == "LocalDiskBackend"
    assert new_position[1] == "LocalCPUBackend"
    assert do_copy
    
    # TODO(Jiayi): We need to align prefetch and move.
    raise NotImplementedError(
        "Prefetch from controller is not implemented yet."
    )
```

#### 2. 自动Eviction迁移 - ❌ 未实现

**当前行为**：
- LocalCPUBackend满了：只删除，不迁移到Disk
- LocalDiskBackend满了：只删除文件，不迁移到Remote

**设计原因**：
- 每个backend独立管理
- 避免不必要的迁移开销
- 保持系统简单

### 4.3 Move功能的限制和控制方式

#### 为什么move只支持从disk到cpu？

**跨节点移动限制**：
- 代码位置：`lmcache/v1/cache_controller/worker.py:413-415`
- **只支持移动到`LocalCPUBackend`**，不支持移动到其他backend

**原因分析**：
1. **P2PBackend的设计**：跨节点移动使用`P2PBackend`进行传输，目标端接收逻辑可能只支持将数据写入CPU内存
2. **初始实现范围**：优先支持最常见的场景（跨节点移动到CPU以加速访问）
3. **技术限制**：P2P传输需要目标端预先分配内存，而CPU内存是最容易分配和管理的

**同一节点内移动限制**：
- 代码框架只支持从`LocalDiskBackend`移动到`LocalCPUBackend`
- **但实际实现是`NotImplementedError`，即功能尚未实现**

#### move是由用户控制的还是程序自动控制的？

**答案**：**move是由用户通过API手动控制的，不是程序自动触发的**

**证据**：
1. **move是一个HTTP API端点**（`api_server/__main__.py:324`），需要通过外部API调用
2. **没有自动触发move的机制**：没有在eviction、lookup等场景自动move
3. **自动机制是prefetch，不是move**：prefetch在lookup时自动触发，但只预取到CPU

**设计原因**：
1. **明确的控制权**：move操作通常涉及跨节点或跨backend的数据迁移，需要用户明确知道何时、何地、移动哪些数据
2. **性能考虑**：move操作可能涉及大量数据传输，需要用户根据实际需求决定是否执行
3. **灵活性**：用户可以根据业务需求自定义move策略

**自动机制 vs 手动控制**：

| 机制 | 触发方式 | 用途 | 代码位置 |
|------|---------|------|---------|
| **自动缓存** | 自动（读取时） | Disk/Remote → CPU | `storage_manager.py:436` |
| **Prefetch** | 自动（lookup时） | 异步预取到CPU | `cache_engine.py:1089` |
| **Move** | 手动（API调用） | 跨节点/跨backend移动 | `api_server/__main__.py:324` |

### 4.4 间接实现移动的方法

#### 方法1：通过get() + batched_put()

**原理**：利用自动缓存机制 + 手动存储

```python
# 1. 从old_backend读取（会自动缓存到CPU）
memory_obj = storage_manager.get(key, location="LocalDiskBackend")

# 2. 手动写入new_backend
storage_manager.batched_put(
    keys=[key],
    memory_objs=[memory_obj],
    location="RemoteBackend"  # 指定新位置
)

# 3. 可选：删除old_backend
storage_manager.batched_remove(keys=[key], locations=["LocalDiskBackend"])
```

#### 方法2：通过Controller API的move（跨节点）

**适用场景**：
- 跨LMCache实例移动
- 需要Controller支持

**限制**：
- 只支持跨节点，不支持同一节点内不同backend之间
- 需要P2PBackend和NIXL

### 4.5 当前能力总结

| 移动类型 | 状态 | 代码位置 | 说明 |
|---------|------|---------|------|
| **跨节点移动** | ✅ 已实现 | `cache_engine.py:1021` | 通过P2PBackend，需要Controller |
| **Disk → CPU（自动）** | ✅ 已实现 | `storage_manager.py:436` | 读取时自动缓存 |
| **Remote → CPU（自动）** | ✅ 已实现 | `storage_manager.py:436` | 读取时自动缓存 |
| **CPU → Disk（手动）** | ✅ 已实现 | `storage_manager.py:374` | 通过location参数 |
| **CPU → Remote（手动）** | ✅ 已实现 | `storage_manager.py:374` | 通过location参数 |
| **同一节点内移动** | ❌ 未实现 | `worker.py:409` | NotImplementedError |
| **自动eviction迁移** | ❌ 未实现 | - | 设计上不支持 |

---

## 总结

### 核心设计原则

1. **分层架构**：Scheduler端负责查找和调度，Worker端负责加载和保存
2. **Prefix Matching**：只返回连续前缀的命中数量，确保一致性
3. **Backend独立性**：每个backend独立管理，没有自动eviction迁移
4. **自动缓存机制**：从慢速backend读取时自动缓存到CPU
5. **手动控制move**：跨节点/跨backend移动需要用户通过API手动控制

### 关键代码位置

- **请求流程**：`lmcache/integration/vllm/vllm_v1_adapter.py`
- **Lookup Client**：`lmcache/v1/lookup_client/`
- **Lookup Engine**：`lmcache/v1/cache_engine.py:887-1016`
- **Storage Manager**：`lmcache/v1/storage_backend/storage_manager.py`
- **Backend创建**：`lmcache/v1/storage_backend/__init__.py:106-230`
- **Move功能**：`lmcache/v1/cache_engine.py:1021-1085`

### 未来改进方向

根据代码中的TODO注释，未来可能会：
1. 实现同一节点内的move（intra-node move）
2. 支持移动到其他backend（不仅仅是CPU）
3. 与prefetch功能对齐，提供更统一的接口

