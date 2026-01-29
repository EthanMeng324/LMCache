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
sudo -E env CUDA_VISIBLE_DEVICES=0 \
  LMCACHE_CONFIG_FILE=examples/cache_migration/migration_test.yaml \
  ./venv/bin/vllm serve Qwen/Qwen2.5-7B-Instruct \
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

python3 benchmarks/multi_round_qa/multi-round-qa.py \
  --num-users 30 \
  --total-users 30 \
  --num-rounds 3 \
  --qps 2 \
  --shared-system-prompt 10000 \
  --user-history-prompt 20000 \
  --answer-len 100 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --base-url http://localhost:8000/v1

输出diff
(
  git diff -- . ':(exclude)*.md' ':(exclude)*.o'
  echo
  git ls-files --others --exclude-standard \
    | grep -vE '\.(md|o)$' \
    | while read f; do
        if [[ "$f" != *.* && -x "$f" ]]; then
          continue
        fi
        echo "===== NEW FILE: $f ====="
        cat "$f"
        echo
      done
) > changes.txt

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



(EngineCore_DP0 pid=772517) [2026-01-27 01:40:09,402] LMCache ERROR: Request chatcmpl-a45fe9554f56f11bThe number of retrieved tokens is less than the expected number of tokens! This should not happen! (vllm_v1_adapter.py:1032:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:09,402] LMCache ERROR: Num retrieved tokens: 0, num expected tokens: 32 (vllm_v1_adapter.py:1038:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:09,403] LMCache WARNING: Request chatcmpl-a45fe9554f56f11b failed to load 256 tokens across 16 blocks (vllm_v1_adapter.py:1125:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:09,923] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:10,071] LMCache INFO: Reqid: chatcmpl-a45fe9554f56f11b, Total tokens 11095, LMCache hit tokens: 4096, need to load: 0 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(APIServer pid=772316) INFO:     127.0.0.1:48048 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:10,182] LMCache INFO: Reqid: chatcmpl-a45fe9554f56f11b, Total tokens 11095, LMCache hit tokens: 4096, need to load: 240 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(APIServer pid=772316) INFO:     127.0.0.1:49854 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:10,294] LMCache INFO: Reqid: chatcmpl-a45fe9554f56f11b, Total tokens 11095, LMCache hit tokens: 4096, need to load: 400 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:10,401] LMCache INFO: Reqid: chatcmpl-a45fe9554f56f11b, Total tokens 11095, LMCache hit tokens: 4096, need to load: 608 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(APIServer pid=772316) INFO:     127.0.0.1:33856 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:10,513] LMCache INFO: Reqid: chatcmpl-a45fe9554f56f11b, Total tokens 11095, LMCache hit tokens: 4096, need to load: 880 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:10,620] LMCache INFO: Reqid: chatcmpl-a45fe9554f56f11b, Total tokens 11095, LMCache hit tokens: 4096, need to load: 1056 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(APIServer pid=772316) INFO:     127.0.0.1:50322 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:10,732] LMCache INFO: Reqid: chatcmpl-a45fe9554f56f11b, Total tokens 11095, LMCache hit tokens: 4096, need to load: 1296 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:10,839] LMCache INFO: Reqid: chatcmpl-a45fe9554f56f11b, Total tokens 11095, LMCache hit tokens: 4096, need to load: 1568 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:10,946] LMCache INFO: Reqid: chatcmpl-a45fe9554f56f11b, Total tokens 11095, LMCache hit tokens: 4096, need to load: 1808 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(APIServer pid=772316) INFO:     127.0.0.1:50432 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:55480 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:11,060] LMCache INFO: Reqid: chatcmpl-a45fe9554f56f11b, Total tokens 11095, LMCache hit tokens: 4096, need to load: 1984 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(APIServer pid=772316) INFO:     127.0.0.1:59282 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:54326 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:48650 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:11,073] LMCache WARNING: The cache block is in the storage, but it can't be retrieved (cache_engine.py:1471:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:11,073] LMCache INFO: Retrieved 0 out of 2048 required tokens (from 4096 total tokens). size: 0.0000 gb, cost 0.4065 ms, throughput: 0.0000 GB/s; (cache_engine.py:701:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:11,073] LMCache ERROR: Request chatcmpl-a45fe9554f56f11bThe number of retrieved tokens is less than the expected number of tokens! This should not happen! (vllm_v1_adapter.py:1032:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:11,073] LMCache ERROR: Num retrieved tokens: 0, num expected tokens: 1984 (vllm_v1_adapter.py:1038:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:11,073] LMCache WARNING: Request chatcmpl-a45fe9554f56f11b failed to load 2048 tokens across 128 blocks (vllm_v1_adapter.py:1125:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:11,595] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:12,170] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(APIServer pid=772316) INFO:     127.0.0.1:57720 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:12,759] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:12,791] LMCache INFO: Reqid: chatcmpl-94b3a38ae4929d68, Total tokens 839, LMCache hit tokens: 512, need to load: -240 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:12,792] LMCache INFO: Reqid: chatcmpl-b8bab01a78e7e446, Total tokens 211, LMCache hit tokens: 0, need to load: -64 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:12,794] LMCache INFO: Reqid: chatcmpl-8a60c111b2b85690, Total tokens 1067, LMCache hit tokens: 512, need to load: -240 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(APIServer pid=772316) INFO:     127.0.0.1:50522 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:13,341] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:13,341] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:13,342] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:13,342] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:13,374] LMCache INFO: Reqid: chatcmpl-949dee28015ff39f, Total tokens 492, LMCache hit tokens: 0, need to load: -144 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(APIServer pid=772316) INFO:     127.0.0.1:40002 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:13,716] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:13,862] LMCache INFO: Reqid: chatcmpl-8a60c111b2b85690, Total tokens 1068, LMCache hit tokens: 512, need to load: -240 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:14,192] LMCache INFO: Reqid: chatcmpl-a45fe9554f56f11b, Total tokens 11100, LMCache hit tokens: 10752, need to load: -16 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:14,194] LMCache INFO: Reqid: chatcmpl-94b3a38ae4929d68, Total tokens 843, LMCache hit tokens: 512, need to load: -240 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(APIServer pid=772316) INFO:     127.0.0.1:56876 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:45332 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:14,196] LMCache INFO: Reqid: chatcmpl-b8bab01a78e7e446, Total tokens 215, LMCache hit tokens: 0, need to load: -64 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:14,196] LMCache INFO: Reqid: chatcmpl-8a60c111b2b85690, Total tokens 1068, LMCache hit tokens: 512, need to load: -240 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:14,197] LMCache INFO: Reqid: chatcmpl-949dee28015ff39f, Total tokens 492, LMCache hit tokens: 0, need to load: -144 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:14,198] LMCache INFO: Reqid: chatcmpl-b2d89a3364f0e629, Total tokens 102, LMCache hit tokens: 0, need to load: -64 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:14,200] LMCache INFO: Reqid: chatcmpl-8242e617e52ad5d9, Total tokens 106, LMCache hit tokens: 0, need to load: -64 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:14,202] LMCache INFO: Reqid: chatcmpl-a55bdba21fa265a4, Total tokens 413, LMCache hit tokens: 0, need to load: -64 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:14,203] LMCache INFO: Reqid: chatcmpl-85d9bd2150b94714, Total tokens 363, LMCache hit tokens: 256, need to load: -64 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:14,205] LMCache INFO: Reqid: chatcmpl-b951c31599dec62c, Total tokens 102, LMCache hit tokens: 0, need to load: -64 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:14,207] LMCache INFO: Reqid: chatcmpl-bd9be5cda72af6f5, Total tokens 146, LMCache hit tokens: 0, need to load: -64 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:14,209] LMCache INFO: Reqid: chatcmpl-a867c905f6045a62, Total tokens 1545, LMCache hit tokens: 1024, need to load: -80 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:14,729] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:14,729] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:14,730] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:14,730] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:14,730] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:14,730] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:14,730] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:14,768] LMCache INFO: Reqid: chatcmpl-b1a46a73666ce022, Total tokens 1858, LMCache hit tokens: 0, need to load: -160 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(APIServer pid=772316) INFO:     127.0.0.1:50356 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:43138 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:15,131] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:15,278] LMCache INFO: Reqid: chatcmpl-a867c905f6045a62, Total tokens 1546, LMCache hit tokens: 1024, need to load: -192 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:15,497] LMCache INFO: Reqid: chatcmpl-b951c31599dec62c, Total tokens 106, LMCache hit tokens: 0, need to load: -64 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:15,498] LMCache INFO: Reqid: chatcmpl-bd9be5cda72af6f5, Total tokens 150, LMCache hit tokens: 0, need to load: -64 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:15,499] LMCache INFO: Reqid: chatcmpl-a867c905f6045a62, Total tokens 1546, LMCache hit tokens: 1024, need to load: -80 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(APIServer pid=772316) INFO:     127.0.0.1:45304 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:15,846] LMCache INFO: Reqid: chatcmpl-a867c905f6045a62, Total tokens 1546, LMCache hit tokens: 1024, need to load: -80 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:15,846] LMCache INFO: Reqid: chatcmpl-b1a46a73666ce022, Total tokens 1858, LMCache hit tokens: 0, need to load: -160 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(APIServer pid=772316) INFO:     127.0.0.1:47042 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:16,210] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:16,582] LMCache INFO: Reqid: chatcmpl-a55bdba21fa265a4, Total tokens 422, LMCache hit tokens: 256, need to load: -48 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:16,687] LMCache INFO: Reqid: chatcmpl-a55bdba21fa265a4, Total tokens 422, LMCache hit tokens: 256, need to load: 160 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:16,793] LMCache INFO: Reqid: chatcmpl-a55bdba21fa265a4, Total tokens 422, LMCache hit tokens: 256, need to load: 192 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(APIServer pid=772316) INFO:     127.0.0.1:38960 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:16,904] LMCache INFO: Reqid: chatcmpl-a55bdba21fa265a4, Total tokens 422, LMCache hit tokens: 256, need to load: 192 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:16,907] LMCache INFO: Reqid: chatcmpl-85d9bd2150b94714, Total tokens 372, LMCache hit tokens: 256, need to load: -64 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:16,909] LMCache INFO: Reqid: chatcmpl-b951c31599dec62c, Total tokens 110, LMCache hit tokens: 0, need to load: -64 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(APIServer pid=772316) INFO:     127.0.0.1:49840 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:16,911] LMCache INFO: Reqid: chatcmpl-bd9be5cda72af6f5, Total tokens 153, LMCache hit tokens: 0, need to load: -64 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:16,913] LMCache INFO: Reqid: chatcmpl-a867c905f6045a62, Total tokens 1547, LMCache hit tokens: 1024, need to load: -80 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:16,922] LMCache WARNING: The cache block is in the storage, but it can't be retrieved (cache_engine.py:1471:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:16,922] LMCache INFO: Retrieved 0 out of 256 required tokens (from 256 total tokens). size: 0.0000 gb, cost 0.2661 ms, throughput: 0.0000 GB/s; (cache_engine.py:701:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:16,922] LMCache ERROR: Request chatcmpl-a55bdba21fa265a4The number of retrieved tokens is less than the expected number of tokens! This should not happen! (vllm_v1_adapter.py:1032:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:16,922] LMCache ERROR: Num retrieved tokens: 0, num expected tokens: 192 (vllm_v1_adapter.py:1038:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:40:16,922] LMCache WARNING: Request chatcmpl-a55bdba21fa265a4 failed to load 256 tokens across 16 blocks (vllm_v1_adapter.py:1125:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) WARNING 01-27 01:40:17 [scheduler.py:1816] Recovered from KV load failure: 215 request(s) rescheduled (373752 tokens affected).
(APIServer pid=772316) INFO 01-27 01:40:18 [loggers.py:248] Engine 000: Avg prompt throughput: 3061.4 tokens/s, Avg generation throughput: 877.0 tokens/s, Running: 215 reqs, Waiting: 47 reqs, GPU KV cache usage: 100.0%, Prefix cache hit rate: 32.8%, External prefix cache hit rate: 0.4%
(APIServer pid=772316) INFO:     127.0.0.1:49560 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:50340 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO 01-27 01:40:28 [loggers.py:248] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 34.9 tokens/s, Running: 212 reqs, Waiting: 70 reqs, GPU KV cache usage: 98.7%, Prefix cache hit rate: 32.8%, External prefix cache hit rate: 0.4%
(APIServer pid=772316) INFO:     127.0.0.1:49516 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:57574 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:44572 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:49502 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:58500 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO 01-27 01:40:38 [loggers.py:248] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 93.2 tokens/s, Running: 207 reqs, Waiting: 95 reqs, GPU KV cache usage: 97.0%, Prefix cache hit rate: 32.8%, External prefix cache hit rate: 0.4%
(APIServer pid=772316) INFO:     127.0.0.1:56920 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:40004 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:38980 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:39030 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:57714 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO 01-27 01:40:48 [loggers.py:248] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 156.7 tokens/s, Running: 202 reqs, Waiting: 122 reqs, GPU KV cache usage: 94.9%, Prefix cache hit rate: 32.8%, External prefix cache hit rate: 0.4%
(APIServer pid=772316) INFO:     127.0.0.1:45424 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:48814 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:48732 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:57698 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO 01-27 01:40:58 [loggers.py:248] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 203.4 tokens/s, Running: 198 reqs, Waiting: 150 reqs, GPU KV cache usage: 92.7%, Prefix cache hit rate: 32.8%, External prefix cache hit rate: 0.4%
(APIServer pid=772316) INFO:     127.0.0.1:48848 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:38990 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:52496 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:45406 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:52450 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:54294 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:46428 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:44628 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:58420 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:43154 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO 01-27 01:41:08 [loggers.py:248] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 249.4 tokens/s, Running: 188 reqs, Waiting: 191 reqs, GPU KV cache usage: 91.6%, Prefix cache hit rate: 32.8%, External prefix cache hit rate: 0.4%
(APIServer pid=772316) INFO:     127.0.0.1:57672 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:55428 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:56888 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:39956 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:45318 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:57674 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:49522 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:44930 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:54280 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:45452 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:49548 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:45464 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO 01-27 01:41:18 [loggers.py:248] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 299.9 tokens/s, Running: 176 reqs, Waiting: 210 reqs, GPU KV cache usage: 85.6%, Prefix cache hit rate: 32.8%, External prefix cache hit rate: 0.4%
(APIServer pid=772316) INFO:     127.0.0.1:43008 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:46994 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:45340 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:43070 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:45498 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:33828 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:54308 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:39962 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO 01-27 01:41:28 [loggers.py:248] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 294.0 tokens/s, Running: 168 reqs, Waiting: 233 reqs, GPU KV cache usage: 82.4%, Prefix cache hit rate: 32.8%, External prefix cache hit rate: 0.4%
(APIServer pid=772316) INFO:     127.0.0.1:48078 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:43122 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:48710 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:45368 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:50446 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:45478 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:32,089] LMCache INFO: Reqid: chatcmpl-bd9be5cda72af6f5, Total tokens 153, LMCache hit tokens: 0, need to load: -64 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:32,089] LMCache INFO: Reqid: chatcmpl-a867c905f6045a62, Total tokens 1547, LMCache hit tokens: 1024, need to load: -80 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(APIServer pid=772316) INFO:     127.0.0.1:45292 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:55440 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:48714 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:32,546] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:32,569] LMCache INFO: Reqid: chatcmpl-b1a46a73666ce022, Total tokens 1858, LMCache hit tokens: 0, need to load: -160 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:32,572] LMCache INFO: Reqid: chatcmpl-b2297ece56daa5f1, Total tokens 3218, LMCache hit tokens: 512, need to load: -240 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:33,033] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:33,033] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:33,033] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:33,520] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:33,550] LMCache INFO: Reqid: chatcmpl-b6bfe3ae1e22472c, Total tokens 915, LMCache hit tokens: 0, need to load: -240 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:33,552] LMCache INFO: Reqid: chatcmpl-ac52425e92324c15, Total tokens 1293, LMCache hit tokens: 0, need to load: -16 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:34,014] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:34,014] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:34,015] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:34,046] LMCache INFO: Reqid: chatcmpl-b206df7e839f1354, Total tokens 949, LMCache hit tokens: 256, need to load: -240 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:34,047] LMCache INFO: Reqid: chatcmpl-8a541ec7e193ac0f, Total tokens 107, LMCache hit tokens: 0, need to load: -64 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:34,049] LMCache INFO: Reqid: chatcmpl-ada2fe4de81a097f, Total tokens 1713, LMCache hit tokens: 1280, need to load: -32 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(APIServer pid=772316) INFO:     127.0.0.1:48624 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:34,051] LMCache INFO: Reqid: chatcmpl-8d9e3f1737dceaf1, Total tokens 1421, LMCache hit tokens: 512, need to load: -224 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:34,514] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:34,515] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:34,515] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:34,515] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:34,515] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:34,543] LMCache INFO: Reqid: chatcmpl-80259f969173aa63, Total tokens 784, LMCache hit tokens: 512, need to load: -224 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:34,544] LMCache INFO: Reqid: chatcmpl-bd8aa548bf303427, Total tokens 935, LMCache hit tokens: 512, need to load: -224 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:34,547] LMCache INFO: Reqid: chatcmpl-b4e453fcda34a110, Total tokens 8323, LMCache hit tokens: 0, need to load: -80 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:35,012] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:35,012] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:35,012] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:35,012] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:35,508] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(APIServer pid=772316) INFO:     127.0.0.1:57544 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:36,016] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(APIServer pid=772316) INFO:     127.0.0.1:48588 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:36,536] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:36,568] LMCache INFO: Reqid: chatcmpl-b5e7ca64e6b7743b, Total tokens 14669, LMCache hit tokens: 0, need to load: -80 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:37,051] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:37,051] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:37,541] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(APIServer pid=772316) INFO:     127.0.0.1:48116 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:50382 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:38,042] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(APIServer pid=772316) INFO:     127.0.0.1:58464 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:57684 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO 01-27 01:41:38 [loggers.py:248] Engine 000: Avg prompt throughput: 2151.6 tokens/s, Avg generation throughput: 338.4 tokens/s, Running: 166 reqs, Waiting: 250 reqs, GPU KV cache usage: 83.6%, Prefix cache hit rate: 32.3%, External prefix cache hit rate: 0.4%
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:38,554] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:39,077] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:39,610] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(APIServer pid=772316) INFO:     127.0.0.1:46474 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:40,163] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(APIServer pid=772316) INFO:     127.0.0.1:43092 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:40,734] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:40,766] LMCache INFO: Reqid: chatcmpl-85c6c8bab39f996c, Total tokens 129, LMCache hit tokens: 0, need to load: -64 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:40,768] LMCache INFO: Reqid: chatcmpl-907482c5bf5691a1, Total tokens 3817, LMCache hit tokens: 3328, need to load: -48 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:40,770] LMCache INFO: Reqid: chatcmpl-beb7979fd70b7278, Total tokens 3151, LMCache hit tokens: 512, need to load: -240 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:41,264] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:41,265] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:41,265] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:41,265] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:41,299] LMCache INFO: Reqid: chatcmpl-9695bcea93addee8, Total tokens 391, LMCache hit tokens: 256, need to load: -80 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:41,778] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:41,810] LMCache INFO: Reqid: chatcmpl-b51796f8323bddfc, Total tokens 434, LMCache hit tokens: 256, need to load: -80 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:41,812] LMCache INFO: Reqid: chatcmpl-8efec4e6b3862dfa, Total tokens 573, LMCache hit tokens: 0, need to load: -80 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:41,814] LMCache INFO: Reqid: chatcmpl-a24ab19b8bab004d, Total tokens 396, LMCache hit tokens: 0, need to load: -64 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:41,816] LMCache INFO: Reqid: chatcmpl-ad168ebf8e32fde9, Total tokens 1853, LMCache hit tokens: 1280, need to load: -32 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:41,818] LMCache INFO: Reqid: chatcmpl-a58bd6a17fd43245, Total tokens 819, LMCache hit tokens: 512, need to load: -240 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:41,819] LMCache INFO: Reqid: chatcmpl-958b73b8d59f753c, Total tokens 985, LMCache hit tokens: 512, need to load: -224 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:41,821] LMCache INFO: Reqid: chatcmpl-b3ae51e76aa00edd, Total tokens 847, LMCache hit tokens: 256, need to load: -240 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:42,298] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:42,298] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:42,298] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:42,298] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:42,299] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:42,299] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:42,330] LMCache INFO: Reqid: chatcmpl-9f99bcefdb6b94a2, Total tokens 795, LMCache hit tokens: 0, need to load: -64 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:42,332] LMCache INFO: Reqid: chatcmpl-b5e23300af072010, Total tokens 106, LMCache hit tokens: 0, need to load: -64 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(APIServer pid=772316) INFO:     127.0.0.1:50424 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:42,334] LMCache INFO: Reqid: chatcmpl-8737a441faad7348, Total tokens 106, LMCache hit tokens: 0, need to load: -64 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:42,336] LMCache INFO: Reqid: chatcmpl-80daa643d61095e0, Total tokens 1407, LMCache hit tokens: 512, need to load: -16 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:42,815] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:42,815] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:42,815] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:42,815] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:42,816] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:42,851] LMCache INFO: Reqid: chatcmpl-b35531f9820bb277, Total tokens 2755, LMCache hit tokens: 512, need to load: -224 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(APIServer pid=772316) INFO:     127.0.0.1:39014 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:43,331] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:43,362] LMCache INFO: Reqid: chatcmpl-894b2ab1fe3a864f, Total tokens 2306, LMCache hit tokens: 512, need to load: -224 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:43,365] LMCache INFO: Reqid: chatcmpl-a6bc98bf8549c1b5, Total tokens 6472, LMCache hit tokens: 0, need to load: -80 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:43,846] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:43,847] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(APIServer pid=772316) INFO:     127.0.0.1:54288 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:33490 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:44,353] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(APIServer pid=772316) INFO:     127.0.0.1:58486 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:44,864] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:45,380] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:45,412] LMCache INFO: Reqid: chatcmpl-9bed8f7d45a4a1b9, Total tokens 6285, LMCache hit tokens: 512, need to load: -240 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:45,890] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:45,890] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:46,397] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(APIServer pid=772316) INFO:     127.0.0.1:48690 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:46,916] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:46,944] LMCache INFO: Reqid: chatcmpl-a16aeae15f81866a, Total tokens 218, LMCache hit tokens: 0, need to load: -80 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:46,946] LMCache INFO: Reqid: chatcmpl-892a166784a52fca, Total tokens 1676, LMCache hit tokens: 0, need to load: -64 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:47,427] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:47,427] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:47,428] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:47,456] LMCache INFO: Reqid: chatcmpl-b8c99af743b54bf2, Total tokens 899, LMCache hit tokens: 512, need to load: 0 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:47,458] LMCache INFO: Reqid: chatcmpl-b0073af9b4c532ac, Total tokens 113, LMCache hit tokens: 0, need to load: -64 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:47,460] LMCache INFO: Reqid: chatcmpl-8b6543c7037af31c, Total tokens 3139, LMCache hit tokens: 512, need to load: -224 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:47,939] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:47,939] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:47,939] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:47,939] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:47,968] LMCache INFO: Reqid: chatcmpl-aec0c4ed7e4ea2e9, Total tokens 2340, LMCache hit tokens: 0, need to load: -64 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(APIServer pid=772316) INFO 01-27 01:41:48 [loggers.py:248] Engine 000: Avg prompt throughput: 5120.2 tokens/s, Avg generation throughput: 328.0 tokens/s, Running: 182 reqs, Waiting: 256 reqs, GPU KV cache usage: 87.9%, Prefix cache hit rate: 32.3%, External prefix cache hit rate: 0.3%
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:48,449] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:48,450] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:48,968] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:48,997] LMCache INFO: Reqid: chatcmpl-bddb1ec43816ca27, Total tokens 2880, LMCache hit tokens: 2048, need to load: -128 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:48,999] LMCache INFO: Reqid: chatcmpl-84250c2e7007fe3b, Total tokens 1599, LMCache hit tokens: 768, need to load: -208 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:49,001] LMCache INFO: Reqid: chatcmpl-8b4c6cc9693f3051, Total tokens 105, LMCache hit tokens: 0, need to load: -64 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:49,003] LMCache INFO: Reqid: chatcmpl-9aa8c174208489df, Total tokens 2287, LMCache hit tokens: 1280, need to load: -32 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:49,489] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:49,490] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:49,490] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:49,490] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:49,523] LMCache INFO: Reqid: chatcmpl-ac44cb9619535d0a, Total tokens 628, LMCache hit tokens: 0, need to load: -64 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:49,525] LMCache INFO: Reqid: chatcmpl-8ae71e2496613466, Total tokens 391, LMCache hit tokens: 256, need to load: -112 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:49,527] LMCache INFO: Reqid: chatcmpl-85d84f711489682f, Total tokens 401, LMCache hit tokens: 256, need to load: -80 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:49,528] LMCache INFO: Reqid: chatcmpl-8774d75b6e5deb95, Total tokens 183, LMCache hit tokens: 0, need to load: -64 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(APIServer pid=772316) INFO:     127.0.0.1:43178 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:49,530] LMCache INFO: Reqid: chatcmpl-a282d06ed64d0719, Total tokens 1401, LMCache hit tokens: 512, need to load: -16 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:50,015] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:50,016] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:50,016] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:50,016] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:50,047] LMCache INFO: Reqid: chatcmpl-8aff08e2c10b24ad, Total tokens 1993, LMCache hit tokens: 1024, need to load: -128 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:50,049] LMCache INFO: Reqid: chatcmpl-b4b541856ae096d8, Total tokens 2809, LMCache hit tokens: 1280, need to load: -32 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:50,538] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:50,538] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:50,539] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:50,568] LMCache INFO: Reqid: chatcmpl-974ca24e86985f2f, Total tokens 851, LMCache hit tokens: 0, need to load: -240 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:50,570] LMCache INFO: Reqid: chatcmpl-ab10f9d8e383d55e, Total tokens 426, LMCache hit tokens: 256, need to load: -80 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:50,572] LMCache INFO: Reqid: chatcmpl-9d23a0bac73b4551, Total tokens 1870, LMCache hit tokens: 0, need to load: -80 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(APIServer pid=772316) INFO:     127.0.0.1:50440 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:51,061] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:51,061] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:51,062] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:51,093] LMCache INFO: Reqid: chatcmpl-a100d5535c1cf801, Total tokens 8919, LMCache hit tokens: 0, need to load: -64 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:51,583] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:51,584] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(APIServer pid=772316) INFO:     127.0.0.1:45500 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:58432 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:52,115] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:52,643] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:53,180] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(APIServer pid=772316) INFO:     127.0.0.1:58400 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:53,732] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:53,766] LMCache INFO: Reqid: chatcmpl-960e108479a1a1cb, Total tokens 2835, LMCache hit tokens: 1024, need to load: -240 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:54,273] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:54,273] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:54,303] LMCache INFO: Reqid: chatcmpl-ac8f8522e3b2b324, Total tokens 120, LMCache hit tokens: 0, need to load: -64 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:54,305] LMCache INFO: Reqid: chatcmpl-adfe8f5a39abecfa, Total tokens 572, LMCache hit tokens: 0, need to load: -64 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:54,306] LMCache INFO: Reqid: chatcmpl-a4ad2b46977ea4be, Total tokens 11210, LMCache hit tokens: 0, need to load: -80 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(APIServer pid=772316) INFO:     127.0.0.1:39528 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:54,798] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:54,798] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:54,798] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:54,798] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:55,315] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(APIServer pid=772316) INFO:     127.0.0.1:58414 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:55,848] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:56,389] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:56,940] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:57,505] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:57,540] LMCache INFO: Reqid: chatcmpl-ad39415aa1389c3c, Total tokens 5283, LMCache hit tokens: 0, need to load: -80 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(APIServer pid=772316) INFO:     127.0.0.1:48676 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:58,063] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:58,063] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(APIServer pid=772316) INFO 01-27 01:41:58 [loggers.py:248] Engine 000: Avg prompt throughput: 4694.7 tokens/s, Avg generation throughput: 364.4 tokens/s, Running: 194 reqs, Waiting: 266 reqs, GPU KV cache usage: 95.5%, Prefix cache hit rate: 31.9%, External prefix cache hit rate: 0.3%
(APIServer pid=772316) INFO:     127.0.0.1:49820 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:58,601] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(APIServer pid=772316) INFO:     127.0.0.1:54352 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:48608 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:59,148] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:59,180] LMCache INFO: Reqid: chatcmpl-892aa64853875b80, Total tokens 6542, LMCache hit tokens: 512, need to load: -240 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(APIServer pid=772316) INFO:     127.0.0.1:52446 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:59,685] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:41:59,686] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:00,219] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(APIServer pid=772316) INFO:     127.0.0.1:56882 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:00,761] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:00,791] LMCache INFO: Reqid: chatcmpl-84d0adbe43f0d429, Total tokens 915, LMCache hit tokens: 256, need to load: -240 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:00,793] LMCache INFO: Reqid: chatcmpl-a08c21528e1481c1, Total tokens 1352, LMCache hit tokens: 1280, need to load: -32 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:00,795] LMCache INFO: Reqid: chatcmpl-b0c25499a9e49ca3, Total tokens 848, LMCache hit tokens: 512, need to load: -224 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:00,797] LMCache INFO: Reqid: chatcmpl-8e14af96938a28ba, Total tokens 2319, LMCache hit tokens: 0, need to load: -80 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:01,308] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:01,308] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:01,309] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:01,309] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:01,838] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:01,874] LMCache INFO: Reqid: chatcmpl-b6483e1b8f67f870, Total tokens 1360, LMCache hit tokens: 1280, need to load: -32 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:01,876] LMCache INFO: Reqid: chatcmpl-b604f763344d81e1, Total tokens 419, LMCache hit tokens: 256, need to load: -80 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:01,877] LMCache INFO: Reqid: chatcmpl-bd410b6db539ba93, Total tokens 2457, LMCache hit tokens: 512, need to load: -240 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:02,380] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:02,380] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:02,414] LMCache INFO: Reqid: chatcmpl-883fbc72d6af3dc6, Total tokens 4709, LMCache hit tokens: 0, need to load: -80 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(APIServer pid=772316) INFO:     127.0.0.1:58436 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:02,916] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:02,916] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:03,454] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:03,485] LMCache INFO: Reqid: chatcmpl-afc8124400323065, Total tokens 109, LMCache hit tokens: 0, need to load: -64 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:03,487] LMCache INFO: Reqid: chatcmpl-b46b28ae44476599, Total tokens 250, LMCache hit tokens: 0, need to load: -80 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:03,488] LMCache INFO: Reqid: chatcmpl-ab19892d396d5309, Total tokens 1678, LMCache hit tokens: 1024, need to load: -160 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:03,490] LMCache INFO: Reqid: chatcmpl-afeaf41201e889fe, Total tokens 843, LMCache hit tokens: 0, need to load: -240 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(APIServer pid=772316) INFO:     127.0.0.1:48702 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:04,001] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:04,001] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:04,001] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:04,001] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:04,002] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:04,035] LMCache INFO: Reqid: chatcmpl-8f6d04453a345e51, Total tokens 1023, LMCache hit tokens: 512, need to load: -224 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:04,037] LMCache INFO: Reqid: chatcmpl-936ff12085732df9, Total tokens 775, LMCache hit tokens: 512, need to load: -224 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:04,038] LMCache INFO: Reqid: chatcmpl-a6f50af789c60594, Total tokens 1165, LMCache hit tokens: 768, need to load: 0 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:04,040] LMCache INFO: Reqid: chatcmpl-b108430dcedb780e, Total tokens 380, LMCache hit tokens: 256, need to load: -80 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:04,041] LMCache INFO: Reqid: chatcmpl-ad9d960487d0b737, Total tokens 1804, LMCache hit tokens: 1024, need to load: -128 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:04,546] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:04,547] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:04,547] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:04,547] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:04,547] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:04,578] LMCache INFO: Reqid: chatcmpl-b73e089deac6a6af, Total tokens 9606, LMCache hit tokens: 0, need to load: -80 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:05,085] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:05,085] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:05,627] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:06,178] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(APIServer pid=772316) INFO:     127.0.0.1:45270 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:06,752] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:07,329] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:07,367] LMCache INFO: Reqid: chatcmpl-8d2e48f23a995fda, Total tokens 2481, LMCache hit tokens: 512, need to load: -240 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:07,883] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:07,884] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:07,916] LMCache INFO: Reqid: chatcmpl-84843cc911bfcf48, Total tokens 956, LMCache hit tokens: 512, need to load: -224 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:07,917] LMCache INFO: Reqid: chatcmpl-ba995b474307fd77, Total tokens 528, LMCache hit tokens: 0, need to load: -16 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:07,919] LMCache INFO: Reqid: chatcmpl-bb837c16779d4e8b, Total tokens 1106, LMCache hit tokens: 512, need to load: -240 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:07,921] LMCache INFO: Reqid: chatcmpl-bc6c0e93818b4208, Total tokens 1959, LMCache hit tokens: 0, need to load: -80 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(APIServer pid=772316) INFO 01-27 01:42:08 [loggers.py:248] Engine 000: Avg prompt throughput: 4383.5 tokens/s, Avg generation throughput: 357.4 tokens/s, Running: 207 reqs, Waiting: 269 reqs, GPU KV cache usage: 99.2%, Prefix cache hit rate: 31.8%, External prefix cache hit rate: 0.3%
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:08,435] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:08,435] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:08,435] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:08,435] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:08,435] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:08,576] LMCache INFO: Reqid: chatcmpl-bc6c0e93818b4208, Total tokens 1959, LMCache hit tokens: 256, need to load: -240 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:08,681] LMCache INFO: Reqid: chatcmpl-bc6c0e93818b4208, Total tokens 1959, LMCache hit tokens: 256, need to load: -240 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:08,786] LMCache INFO: Reqid: chatcmpl-bc6c0e93818b4208, Total tokens 1959, LMCache hit tokens: 256, need to load: -240 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:08,892] LMCache INFO: Reqid: chatcmpl-bc6c0e93818b4208, Total tokens 1959, LMCache hit tokens: 256, need to load: -240 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:08,999] LMCache INFO: Reqid: chatcmpl-bc6c0e93818b4208, Total tokens 1959, LMCache hit tokens: 256, need to load: -240 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(APIServer pid=772316) INFO:     127.0.0.1:47168 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:09,108] LMCache INFO: Reqid: chatcmpl-bc6c0e93818b4208, Total tokens 1959, LMCache hit tokens: 256, need to load: -128 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:09,215] LMCache INFO: Reqid: chatcmpl-bc6c0e93818b4208, Total tokens 1959, LMCache hit tokens: 256, need to load: 80 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(APIServer pid=772316) INFO:     127.0.0.1:45390 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:09,326] LMCache INFO: Reqid: chatcmpl-bc6c0e93818b4208, Total tokens 1959, LMCache hit tokens: 256, need to load: 176 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:09,431] LMCache INFO: Reqid: chatcmpl-bc6c0e93818b4208, Total tokens 1959, LMCache hit tokens: 256, need to load: 176 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(APIServer pid=772316) INFO:     127.0.0.1:43056 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:57676 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:09,541] LMCache INFO: Reqid: chatcmpl-bc6c0e93818b4208, Total tokens 1959, LMCache hit tokens: 256, need to load: 176 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:09,647] LMCache INFO: Reqid: chatcmpl-bc6c0e93818b4208, Total tokens 1959, LMCache hit tokens: 256, need to load: 176 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:09,650] LMCache INFO: Reqid: chatcmpl-b4b9b7c53172cf1a, Total tokens 6323, LMCache hit tokens: 0, need to load: -80 (vllm_v1_adapter.py:1550:lmcache.integration.vllm.vllm_v1_adapter)
(APIServer pid=772316) INFO:     127.0.0.1:46452 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:60852 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:56866 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:09,659] LMCache WARNING: The cache block is in the storage, but it can't be retrieved (cache_engine.py:1471:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:09,659] LMCache INFO: Retrieved 0 out of 256 required tokens (from 256 total tokens). size: 0.0000 gb, cost 0.2769 ms, throughput: 0.0000 GB/s; (cache_engine.py:701:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:09,659] LMCache ERROR: Request chatcmpl-bc6c0e93818b4208The number of retrieved tokens is less than the expected number of tokens! This should not happen! (vllm_v1_adapter.py:1032:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:09,659] LMCache ERROR: Num retrieved tokens: 0, num expected tokens: 176 (vllm_v1_adapter.py:1038:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:09,659] LMCache WARNING: Request chatcmpl-bc6c0e93818b4208 failed to load 256 tokens across 16 blocks (vllm_v1_adapter.py:1125:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:42:10,164] LMCache WARNING: Local cpu memory under pressure so choosing to store only  0 total chunks of KV cache. (cache_engine.py:380:lmcache.v1.cache_engine)
(EngineCore_DP0 pid=772517) WARNING 01-27 01:42:10 [scheduler.py:1816] Recovered from KV load failure: 205 request(s) rescheduled (371033 tokens affected).
(APIServer pid=772316) INFO 01-27 01:42:18 [loggers.py:248] Engine 000: Avg prompt throughput: 507.1 tokens/s, Avg generation throughput: 265.6 tokens/s, Running: 205 reqs, Waiting: 291 reqs, GPU KV cache usage: 99.2%, Prefix cache hit rate: 31.7%, External prefix cache hit rate: 0.3%
(APIServer pid=772316) INFO:     127.0.0.1:45378 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:54284 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO 01-27 01:42:28 [loggers.py:248] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 85.3 tokens/s, Running: 203 reqs, Waiting: 321 reqs, GPU KV cache usage: 97.8%, Prefix cache hit rate: 31.7%, External prefix cache hit rate: 0.3%
(APIServer pid=772316) INFO:     127.0.0.1:57664 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:54282 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:50392 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO 01-27 01:42:38 [loggers.py:248] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 170.1 tokens/s, Running: 200 reqs, Waiting: 346 reqs, GPU KV cache usage: 97.2%, Prefix cache hit rate: 31.7%, External prefix cache hit rate: 0.3%
(APIServer pid=772316) INFO:     127.0.0.1:50370 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:58478 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:50466 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:57590 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:54270 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:57532 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO 01-27 01:42:48 [loggers.py:248] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 214.4 tokens/s, Running: 194 reqs, Waiting: 375 reqs, GPU KV cache usage: 96.4%, Prefix cache hit rate: 31.7%, External prefix cache hit rate: 0.3%
(APIServer pid=772316) INFO:     127.0.0.1:46962 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:50398 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:58440 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:43000 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:57618 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO 01-27 01:42:58 [loggers.py:248] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 230.5 tokens/s, Running: 187 reqs, Waiting: 407 reqs, GPU KV cache usage: 95.2%, Prefix cache hit rate: 31.7%, External prefix cache hit rate: 0.3%
(APIServer pid=772316) INFO:     127.0.0.1:48746 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:54316 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:43040 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO 01-27 01:43:08 [loggers.py:248] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 260.4 tokens/s, Running: 182 reqs, Waiting: 428 reqs, GPU KV cache usage: 93.3%, Prefix cache hit rate: 31.7%, External prefix cache hit rate: 0.3%
(APIServer pid=772316) INFO:     127.0.0.1:42998 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:38946 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:48764 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:57690 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:39540 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:39012 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:50426 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO 01-27 01:43:18 [loggers.py:248] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 298.7 tokens/s, Running: 173 reqs, Waiting: 455 reqs, GPU KV cache usage: 91.0%, Prefix cache hit rate: 31.7%, External prefix cache hit rate: 0.3%
(APIServer pid=772316) INFO:     127.0.0.1:39520 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:39574 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO:     127.0.0.1:46444 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=772316) INFO 01-27 01:43:28 [loggers.py:248] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 309.9 tokens/s, Running: 169 reqs, Waiting: 491 reqs, GPU KV cache usage: 91.5%, Prefix cache hit rate: 31.7%, External prefix cache hit rate: 0.3%
(EngineCore_DP0 pid=772517) [2026-01-27 01:43:28,406] LMCache ERROR: The number of tokens is more than the number of blocks for request chatcmpl-bc6c0e93818b4208. Something might be wrong in scheduling logic! (vllm_v1_adapter.py:401:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) [2026-01-27 01:43:28,406] LMCache ERROR: Num tokens: 3813, num blocks: 123, block size: 16 (vllm_v1_adapter.py:407:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [dump_input.py:72] Dumping input data for V1 LLM engine (v0.13.0) with config: model='Qwen/Qwen2.5-7B-Instruct', speculative_config=None, tokenizer='Qwen/Qwen2.5-7B-Instruct', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=32768, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False), seed=0, served_model_name=Qwen/Qwen2.5-7B-Instruct, enable_prefix_caching=True, enable_chunked_prefill=True, pooler_config=None, compilation_config={'level': None, 'mode': <CompilationMode.VLLM_COMPILE: 3>, 'debug_dump_path': None, 'cache_dir': '/home/ucmerced/.cache/vllm/torch_compile_cache/b448d1105a', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['none'], 'splitting_ops': ['vllm::unified_attention', 'vllm::unified_attention_with_output', 'vllm::unified_mla_attention', 'vllm::unified_mla_attention_with_output', 'vllm::mamba_mixer2', 'vllm::mamba_mixer', 'vllm::short_conv', 'vllm::linear_attention', 'vllm::plamo2_mamba_mixer', 'vllm::gdn_attention_core', 'vllm::kda_attention', 'vllm::sparse_attn_indexer'], 'compile_mm_encoder': False, 'compile_sizes': [], 'compile_ranges_split_points': [2048], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>, 'cudagraph_num_of_warmups': 1, 'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': False, 'fuse_act_quant': False, 'fuse_attn_quant': False, 'eliminate_noops': True, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False}, 'max_cudagraph_capture_size': 512, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False}, 'local_cache_dir': '/home/ucmerced/.cache/vllm/torch_compile_cache/b448d1105a/rank_0_0/backbone'}, 
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [dump_input.py:79] Dumping scheduler output for model execution: SchedulerOutput(scheduled_new_reqs=[], scheduled_cached_reqs=CachedRequestData(req_ids=['chatcmpl-9fdc1ccb3b221908', 'chatcmpl-a844dffd67b1b61c', 'chatcmpl-bb0c326d6ad99db3', 'chatcmpl-a2f7eeb55b857d67', 'chatcmpl-9e81b5b2ad6b4bb8', 'chatcmpl-88f1c0d4b814cfc4', 'chatcmpl-8fc9fe99ddbbf981', 'chatcmpl-a874febf8a908c01', 'chat

(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868] EngineCore encountered a fatal error.
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868] Traceback (most recent call last):
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]   File "/home/ucmerced/LMCache/venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 859, in run_engine_core
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]     engine_core.run_busy_loop()
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]   File "/home/ucmerced/LMCache/venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 886, in run_busy_loop
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]     self._process_engine_step()
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]   File "/home/ucmerced/LMCache/venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 919, in _process_engine_step
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]     outputs, model_executed = self.step_fn()
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]                               ^^^^^^^^^^^^^^
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]   File "/home/ucmerced/LMCache/venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 351, in step
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]     model_output = future.result()
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]                    ^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]   File "/home/ucmerced/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/concurrent/futures/_base.py", line 449, in result
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]     return self.__get_result()
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]            ^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]   File "/home/ucmerced/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/concurrent/futures/_base.py", line 401, in __get_result
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]     raise self._exception
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]   File "/home/ucmerced/LMCache/venv/lib/python3.12/site-packages/vllm/v1/executor/uniproc_executor.py", line 79, in collective_rpc
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]     result = run_method(self.driver_worker, method, args, kwargs)
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]   File "/home/ucmerced/LMCache/venv/lib/python3.12/site-packages/vllm/v1/serial_utils.py", line 461, in run_method
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]     return func(*args, **kwargs)
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]   File "/home/ucmerced/LMCache/venv/lib/python3.12/site-packages/vllm/v1/worker/worker_base.py", line 369, in execute_model
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]     return self.worker.execute_model(scheduler_output, *args, **kwargs)
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]   File "/home/ucmerced/LMCache/venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]     return func(*args, **kwargs)
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]   File "/home/ucmerced/LMCache/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py", line 623, in execute_model
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]     output = self.model_runner.execute_model(
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]   File "/home/ucmerced/LMCache/venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]     return func(*args, **kwargs)
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]   File "/home/ucmerced/LMCache/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py", line 3085, in execute_model
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]     self.maybe_get_kv_connector_output(scheduler_output) as kv_connector_output,
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]   File "/home/ucmerced/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/contextlib.py", line 144, in __exit__
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]     next(self.gen)
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]   File "/home/ucmerced/LMCache/venv/lib/python3.12/site-packages/vllm/v1/worker/kv_connector_model_runner_mixin.py", line 133, in _get_kv_connector_output
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]     kv_connector.wait_for_save()
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]   File "/home/ucmerced/LMCache/venv/lib/python3.12/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py", line 171, in wait_for_save
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]     self._lmcache_engine.wait_for_save()
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]   File "/home/ucmerced/LMCache/lmcache/integration/vllm/vllm_v1_adapter.py", line 1299, in wait_for_save
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]     assert len(slot_mapping) == len(token_ids)
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=772517) ERROR 01-27 01:43:28 [core.py:868] AssertionError
(APIServer pid=772316) ERROR 01-27 01:43:28 [async_llm.py:538] AsyncLLM output_handler failed.
(APIServer pid=772316) ERROR 01-27 01:43:28 [async_llm.py:538] Traceback (most recent call last):
(APIServer pid=772316) ERROR 01-27 01:43:28 [async_llm.py:538]   File "/home/ucmerced/LMCache/venv/lib/python3.12/site-packages/vllm/v1/engine/async_llm.py", line 490, in output_handler
(APIServer pid=772316) ERROR 01-27 01:43:28 [async_llm.py:538]     outputs = await engine_core.get_output_async()
(APIServer pid=772316) ERROR 01-27 01:43:28 [async_llm.py:538]               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=772316) ERROR 01-27 01:43:28 [async_llm.py:538]   File "/home/ucmerced/LMCache/venv/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 895, in get_output_async
(APIServer pid=772316) ERROR 01-27 01:43:28 [async_llm.py:538]     raise self._format_exception(outputs) from None
(APIServer pid=772316) ERROR 01-27 01:43:28 [async_llm.py:538] vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue. See stack trace (above) for the root cause.
(APIServer pid=772316) INFO:     127.0.0.1:44618 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:57538 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:48750 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:57556 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:33840 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:44640 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:50402 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:47148 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:39046 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:46502 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:45284 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:45356 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:45514 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:52468 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:52472 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:47162 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:56904 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:45350 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:54248 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:54340 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:48056 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:38976 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:48064 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:46460 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:44914 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:45462 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:48074 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:48086 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:54304 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:33496 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:50408 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:46466 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:50480 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:50496 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:50498 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:50512 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:46958 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:47006 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:47020 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:58534 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:44910 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:50362 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:52514 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:49532 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:42974 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:42982 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:58522 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:43082 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:43094 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:43108 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:46436 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:57542 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:43146 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:57678 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:38966 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:45484 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:43166 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:43172 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:46486 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:39028 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:55426 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:60868 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:46986 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:49786 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:45412 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:47038 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:52464 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:43016 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:43088 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:57682 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:59288 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:56870 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:55448 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:55464 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:39004 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:48598 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:48602 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:47024 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:48656 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:48666 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:50376 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:48100 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:46476 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:48698 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:48730 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:48736 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:48748 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:48756 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:48774 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:48788 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:48796 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:48842 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:48800 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:48802 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:39486 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:39496 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:39940 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:33498 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:46978 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:57660 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:39500 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:39502 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:42978 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:39512 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:39556 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:39024 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:39572 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:45438 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:54262 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:52508 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:39588 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(APIServer pid=772316) INFO:     127.0.0.1:50472 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error