# KV Cache Events Medium 字段问题修复

## 问题描述

在使用 LMCache 与 vLLM 集成时，发现 `BlockStored` 事件的 `medium` 字段始终显示为 `"GPU"`，而不是预期的 `"cpu"`（因为 LMCache 会将 KV cache 卸载到 CPU）。

## 根本原因

问题有两个层面：

### 1. KV Events 未启用

LMCache 的 `enable_kv_events` 配置默认为 `False`，即使 vLLM 配置了 `--kv-events-config`，LMCache 也不会生成 KV events。

相关代码位置：`lmcache/v1/cache_engine.py`

```python
# KV events
self.kv_events_enabled = False
self.kv_events_enabled = config.enable_kv_events
if self.kv_events_enabled:
    self.kv_events: List[CacheStoreEvent] = []
    logger.info("KV events are enabled.")
else:
    logger.info("KV events are disabled.")
```

### 2. 不完整 Chunk 未保存

当请求的 token 数量小于 `chunk_size`（例如 35 < 256）时，如果 `discard_partial_chunks=True`（默认值，当 `save_unfull_chunk=False` 时），则 `num_tokens_to_save = 0`，导致 `token_ids = []`，`wait_for_save()` 跳过保存，因此不会生成 KV events。

相关代码位置：`lmcache/integration/vllm/vllm_v1_adapter.py`

**`discard_partial_chunks` 的计算逻辑**（第763-768行）：

```python
# Whether to discard partial chunks
self._discard_partial_chunks = (
    vllm_config.kv_transfer_config.get_from_extra_config(
        "discard_partial_chunks", False
    )
    or not config.save_unfull_chunk
)
```

当 `save_unfull_chunk=False` 时，`discard_partial_chunks=True`。

**`num_tokens_to_save` 的计算逻辑**（第371-376行）：

```python
if not is_last_prefill or discard_partial_chunks:
    num_tokens_to_save = (
        input_token_len // lmcache_chunk_size * lmcache_chunk_size
    )
else:
    num_tokens_to_save = input_token_len
```

当 `input_token_len=35`，`lmcache_chunk_size=256`，`discard_partial_chunks=True` 时：
- `num_tokens_to_save = 35 // 256 * 256 = 0 * 256 = 0`

**`token_ids` 的截取逻辑**（第384行）：

```python
token_ids = input_token_ids[:num_tokens_to_save]
```

当 `num_tokens_to_save=0` 时，`token_ids = []`。

**`wait_for_save()` 的跳过逻辑**（第1323-1325行）：

```python
if skip_leading_tokens == len(token_ids):
    continue  # skip this request
```

当 `token_ids = []` 且 `skip_leading_tokens = 0` 时，条件满足，跳过保存，因此不会生成 KV events。

## 解决方案

在 LMCache 配置文件中添加以下配置：

```yaml
# Enable KV cache events for debugging
enable_kv_events: true

# Save unfull chunks (chunks smaller than chunk_size)
# This allows saving KV cache even when token count < chunk_size
save_unfull_chunk: true
```

### 配置说明

1. **`enable_kv_events: true`**
   - 启用 LMCache 生成 KV cache events
   - 这是必需的，因为默认情况下 LMCache 不会生成 events

2. **`save_unfull_chunk: true`**
   - 允许保存不完整的 chunk（token 数 < chunk_size）
   - 当设置为 `true` 时，`discard_partial_chunks` 会变为 `False`
   - 这样即使 token 数小于 chunk_size，也会保存 KV cache 并生成 events

## 验证

修复后，日志中应该能看到：

1. **LMCache 创建 event**：
   ```
   Added kv cache event with medium='cpu' to kv cache events queue
   ```

2. **返回给 vLLM 的 event**：
   ```
   LMCacheEngine.get_kv_events: returning 1 events, first event medium='cpu'
   LMCacheConnector.get_kv_events: returning event with medium='cpu'
   ```

3. **最终发布的 event**：
   ```
   BlockStored(..., medium='cpu')
   ```

## 注意事项

- vLLM 自己的 prefix caching 产生的 `BlockStored` events 仍然会显示 `medium='GPU'`，这是正常的
- 只有 LMCache 生成的 events 才会显示 `medium='cpu'`
- 如果看到多个 `BlockStored` events，其中一些是 vLLM 的（GPU），一些是 LMCache 的（CPU）

