# KV 缓存迁移功能使用指南

本功能会自动将访问频率最高的 N 个 KV 缓存项从一个存储后端迁移到另一个存储后端，已直接集成到 LMCache 主程序中。

## 目录

1. [快速开始](#快速开始)
2. [功能说明](#功能说明)
3. [配置说明](#配置说明)
4. [运行与测试](#运行与测试)
5. [验证迁移](#验证迁移)
6. [测试步骤](#测试步骤)
7. [代码使用](#代码使用)
8. [故障排查](#故障排查)

## 快速开始

### 前置要求

- Python 3.8+
- 支持 CUDA 的 GPU（用于 vLLM）
- 已安装 vLLM
- 已安装 LMCache（v0.3.5 分支）
- 至少 10GB 可用磁盘空间（用于磁盘后端）

### 快速测试

1. **创建磁盘缓存目录：**
```bash
mkdir -p /tmp/data/hm/lmcache_test

# 1. 编译 C 库（使用 Makefile，推荐）
make -f Makefile.cxl_shm

# 2. 将库文件复制到项目目录
cp libcxl_shm.so lmcache/v1/storage_backend/
```

2. **使用提供的配置文件：** `migration_test.yaml`

3. **启动 vLLM 与 LMCache：**
```bash
CUDA_VISIBLE_DEVICES=0 \
LMCACHE_CONFIG_FILE=examples/cache_migration/migration_test.yaml \
vllm serve \
  Qwen/Qwen2.5-7B-Instruct \
  --gpu-memory-utilization 0.8 \
  --port 8000 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
  --kv-events-config '{"enable_kv_cache_events": "True", "publisher": "zmq", "topic": "kv-events"}'
```

curl http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "prompt": "<|begin_of_text|><|system|>\nYou are a helpful AI assistant.\n<|user|>\nWhat is the capital of France?\n<|assistant|>",
  "max_tokens": 100,
  "temperature": 0.7
}'

4. **运行测试脚本：**
```bash
cd examples/cache_migration
python test_migration.py
```

## 功能说明

缓存迁移服务已集成到 `StorageManager` 中，在 KV 缓存存储操作期间自动运行。启用后，它会：

1. 监控 KV 缓存项的访问次数（支持 LFU 和 LRU 策略）
2. 定期将访问频率最高的 N 个项从源后端迁移到目标后端
3. 在主程序的调用栈中自动运行（不在单独的线程中）

### 工作原理

迁移服务在 `StorageManager.__init__()` 中自动初始化（如果通过配置启用）。它在 `batched_put()` 操作期间运行：

1. 当通过 `batched_put()` 存储 KV 缓存时
2. 服务检查是否到了迁移时间（基于 `migration_interval`）
3. 如果是，则迁移访问频率最高的 N 个键
4. 所有操作都在主程序的调用栈中执行

### 访问计数跟踪

- **LFU 策略**：使用现有的频率跟踪（`key_to_freq`）
- **LRU 策略**：现在跟踪访问计数（在此实现中添加）
- **其他策略**：会记录警告并返回空统计信息

## 配置说明

### 基本配置

在 LMCache 配置文件（YAML）中添加以下内容：

```yaml
# 基本 LMCache 配置
chunk_size: 256
local_cpu: true
max_local_cpu_size: 2.0  # 2GB CPU 缓存
local_disk: "/tmp/data/hm/lmcache_test"  # 磁盘缓存目录
max_local_disk_size: 10.0  # 10GB 磁盘缓存

# 缓存策略（LRU 或 LFU - 两者都支持访问跟踪）
cache_policy: "LRU"  # 或 "LFU"

# 启用缓存迁移
extra_config:
  # 启用迁移功能
  enable_cache_migration: true
  
  # 源后端（从哪里迁移）
  cache_migration_source_backend: "LocalCPUBackend"
  
  # 目标后端（迁移到哪里）
  cache_migration_target_backend: "LocalDiskBackend"
  
  # 要迁移的访问频率最高的键数量
  cache_migration_top_n: 5
  
  # 迁移之间的最小间隔（秒）
  # 迁移只会在间隔时间过去且存储了新缓存时发生
  cache_migration_interval: 30.0
  
  # 复制模式：如果为 true，在源中保留原始（复制到目标）
  # 如果为 false，移动它（从源删除，添加到目标）
  cache_migration_copy_mode: true
```

### 配置参数说明

- `enable_cache_migration` (bool, 默认: false): 启用/禁用缓存迁移
- `cache_migration_source_backend` (str, 默认: "LocalCPUBackend"): 源后端名称
- `cache_migration_target_backend` (str, 默认: "LocalDiskBackend"): 目标后端名称
- `cache_migration_top_n` (int, 默认: 10): 要迁移的访问频率最高的键数量
- `cache_migration_interval` (float, 默认: 60.0): 迁移之间的最小间隔（秒）
- `cache_migration_copy_mode` (bool, 默认: true): 复制模式（true）或移动模式（false）

### 配置示例

#### 示例 1: 将热缓存从 CPU 迁移到磁盘

```yaml
local_cpu: true
max_local_cpu_size: 10.0  # 10GB
local_disk: "/mnt/cache"
max_local_disk_size: 100.0  # 100GB

extra_config:
  enable_cache_migration: true
  cache_migration_source_backend: "LocalCPUBackend"
  cache_migration_target_backend: "LocalDiskBackend"
  cache_migration_top_n: 20
  cache_migration_interval: 120.0  # 每 2 分钟
  cache_migration_copy_mode: true  # 保留在 CPU，也复制到磁盘
```

#### 示例 2: 将频繁访问的项从磁盘移动到 CPU

```yaml
local_cpu: true
max_local_cpu_size: 10.0
local_disk: "/mnt/cache"
max_local_disk_size: 100.0

extra_config:
  enable_cache_migration: true
  cache_migration_source_backend: "LocalDiskBackend"
  cache_migration_target_backend: "LocalCPUBackend"
  cache_migration_top_n: 10
  cache_migration_interval: 30.0  # 每 30 秒
  cache_migration_copy_mode: false  # 移动（从磁盘删除）
```

## 运行与测试

### 使用 vLLM 运行 LMCache

1. **启动 vLLM 并启用 LMCache：**

```bash
CUDA_VISIBLE_DEVICES=0 \
LMCACHE_CONFIG_FILE=migration_test.yaml \
vllm serve \
  Qwen/Qwen2.5-7B-Instruct \
  --gpu-memory-utilization 0.8 \
  --port 8000 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```

2. **等待 vLLM 启动**（您应该看到表明 LMCache 已初始化的日志）

3. **检查迁移服务初始化的日志：**

查找此日志消息：
```
INFO Cache migration service initialized: source=LocalCPUBackend, target=LocalDiskBackend, top_n=5, interval=30.0s, copy_mode=True
```

### 使用测试脚本

我们提供了一个自动化测试脚本 `test_migration.py`：

```bash
cd examples/cache_migration
python test_migration.py
```

该脚本会：
- 连接到 vLLM 服务器
- 发送请求生成 KV 缓存
- 多次访问以增加访问计数
- 等待迁移间隔
- 提供验证建议

## 验证迁移

### 方法 1: 检查日志

迁移活动在 DEBUG 和 INFO 级别记录。启用调试日志：

```bash
export LMCACHE_LOG_LEVEL=DEBUG
```

查找以下日志消息：

1. **迁移服务已初始化：**
```
INFO Cache migration service initialized: source=LocalCPUBackend, target=LocalDiskBackend, top_n=5, interval=30.0s, copy_mode=True
```

2. **迁移已触发：**
```
DEBUG Migrated 3 keys after batched_put
```

3. **迁移详情（如果启用了调试）：**
```
INFO Migrating top 5 keys from <LocalCPUBackend> to <LocalDiskBackend>
DEBUG Starting migration cycle
```

### 方法 2: 检查后端内容

您可以编程方式检查键是否存在于两个后端中：

```python
from lmcache.v1.cache_engine import LMCacheEngineBuilder

# 获取引擎实例（如果您有访问权限）
engine = LMCacheEngineBuilder.get("your_instance_id")
if engine:
    storage_manager = engine.storage_manager
    
    # 获取后端
    cpu_backend = storage_manager.storage_backends.get("LocalCPUBackend")
    disk_backend = storage_manager.storage_backends.get("LocalDiskBackend")
    
    # 检查 CPU 后端中的键
    if cpu_backend:
        cpu_keys = cpu_backend.get_keys()
        print(f"CPU 后端中的键数量: {len(cpu_keys)}")
    
    # 检查磁盘后端中的键
    if disk_backend:
        disk_keys = disk_backend.get_keys()
        print(f"磁盘后端中的键数量: {len(disk_keys)}")
```

### 方法 3: 监控访问计数

检查访问计数以查看哪些键被跟踪：

```python
from lmcache.v1.storage_backend.cache_access_monitor import CacheAccessMonitor

storage_manager = engine.storage_manager
cpu_backend = storage_manager.storage_backends.get("LocalCPUBackend")

if cpu_backend:
    monitor = CacheAccessMonitor(cpu_backend)
    access_counts = monitor.get_access_counts()
    top_keys = monitor.get_top_keys(top_n=10)
    
    print(f"具有访问跟踪的键总数: {len(access_counts)}")
    print("访问频率最高的 10 个键:")
    for key, count in top_keys:
        print(f"  键 {key}: {count} 次访问")
```

## 测试步骤

### 测试 1: 基本迁移测试

1. **启动启用迁移的 vLLM**（参见运行部分）

2. **发送多个相同提示的请求**以生成缓存命中：

```bash
# 第一个请求（将缓存）
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "prompt": "什么是机器学习？",
    "max_tokens": 50
  }'

# 多次发送相同请求以增加访问计数
for i in {1..10}; do
  curl -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Qwen/Qwen2.5-7B-Instruct",
      "prompt": "什么是机器学习？",
      "max_tokens": 50
    }'
  sleep 1
done
```

3. **等待迁移间隔**（示例配置中为 30 秒）

4. **检查日志**中的迁移消息

5. **验证键是否存在于两个后端**（如果 copy_mode=true）

### 测试 2: 验证访问计数跟踪

1. **发送不同提示的请求**以创建多个缓存条目

2. **更频繁地访问某些提示**而不是其他提示

3. **使用监控脚本检查访问计数**（参见方法 3）

4. **验证频繁访问的键首先被迁移**

### 测试 3: 迁移间隔测试

1. **在配置中设置较短的迁移间隔**（例如 10 秒）

2. **持续发送请求**

3. **监控日志**以验证迁移在配置的间隔发生

4. **检查迁移不会比间隔更频繁地发生**

### 测试 4: 复制 vs 移动模式

1. **使用 copy_mode=true 测试：**
   - 发送请求填充 CPU 缓存
   - 等待迁移
   - 验证键同时存在于 CPU 和磁盘后端

2. **使用 copy_mode=false 测试：**
   - 发送请求填充 CPU 缓存
   - 等待迁移
   - 验证键从 CPU 后端删除，仅存在于磁盘后端

## 代码使用

您不需要编写任何代码 - 它是自动集成的！只需通过配置启用即可。

但是，如果您想手动触发迁移或检查状态：

```python
# 从存储管理器获取迁移服务
storage_manager = lmcache_engine.storage_manager
migration_service = storage_manager.cache_migration_service

if migration_service is not None:
    # 检查是否到了迁移时间
    if migration_service.should_migrate():
        # 触发迁移
        num_migrated = migration_service.migrate()
        print(f"已迁移 {num_migrated} 个键")
```

## 故障排查

### 迁移未发生

1. **检查配置：**
   ```bash
   # 验证 enable_cache_migration 为 true
   grep -A 5 "extra_config" migration_test.yaml
   ```

2. **检查初始化日志：**
   - 查找 "Cache migration service initialized" 消息
   - 如果缺失，检查关于缺失后端的警告

3. **验证后端存在：**
   - 确保配置了 `LocalCPUBackend` 和 `LocalDiskBackend`
   - 检查磁盘目录是否存在且可写

4. **检查缓存策略：**
   - 迁移需要 LRU 或 LFU 策略
   - 其他策略不支持访问跟踪

5. **验证迁移间隔：**
   - 迁移只在调用 `batched_put()` 时发生
   - 如果没有存储新的 KV 缓存，迁移不会触发
   - 确保您发送的请求会生成新缓存

### 无访问计数

1. **检查缓存策略：**
   - 必须是 LRU 或 LFU
   - 其他策略不跟踪访问计数

2. **验证键是否被访问：**
   - 键必须被命中（检索）才能增加访问计数
   - 仅存储键不算作访问

### 迁移失败

1. **检查磁盘空间：**
   ```bash
   df -h /tmp/data/hm/lmcache_test
   ```

2. **检查权限：**
   ```bash
   ls -ld /tmp/data/hm/lmcache_test
   ```

3. **检查日志中的错误：**
   - 查找 "Cache migration failed" 警告
   - 检查权限错误或磁盘已满错误

### 键未迁移

1. **检查 top_n 设置：**
   - 只迁移前 N 个键
   - 如果您的键少于 N 个，所有键都会被迁移
   - 如果您的键多于 N 个，只有访问计数最高的前 N 个会被迁移

2. **验证访问计数：**
   - 访问次数为 0 的键不会进入前 N 个
   - 确保键被多次访问（命中）

3. **检查迁移间隔：**
   - 迁移只在间隔时间过去后发生
   - 并且只在调用 `batched_put()` 时发生

## 预期行为

当迁移正常工作时，您应该看到：

1. **启动时：**
   - 日志消息："Cache migration service initialized"

2. **运行期间：**
   - 定期日志消息："Migrated X keys after batched_put"
   - 这些在迁移间隔过去且存储了新缓存时出现

3. **在复制模式下：**
   - 键同时存在于源和目标后端
   - 访问计数继续在源后端跟踪

4. **在移动模式下：**
   - 键从源后端删除
   - 键仅存在于目标后端

## 注意事项

- 迁移在 `batched_put()` 操作期间同步运行
- 它被设计为非阻塞且快速
- 迁移失败会被记录但不会停止主程序
- 对于大型迁移，考虑增加 `migration_interval` 以降低频率

## 下一步

- 监控迁移频率并根据需要调整 `migration_interval`
- 尝试不同的 `top_n` 值
- 使用不同的源/目标后端组合进行测试
- 使用磁盘后端时监控磁盘使用情况
