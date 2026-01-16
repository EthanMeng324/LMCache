# 迁移未提交修改到 v0.3.11

## 快速操作（推荐）

```bash
# 1. 保存所有修改（包括未跟踪文件）
git stash push -u -m "迁移到v0.3.11"

# 2. 切换到 v0.3.11 并创建分支
git checkout -b v0.3.11-branch v0.3.11

# 3. 应用修改
git stash pop
```

## 完整命令说明

| 命令 | 说明 |
|------|------|
| `git stash push -u` | 保存所有修改（`-u` 包含未跟踪文件） |
| `git checkout -b v0.3.11-branch v0.3.11` | 基于标签创建新分支 |
| `git stash pop` | 应用并删除保存的修改 |

## 如果遇到冲突

```bash
# 解决冲突后
git add .
git stash drop  # 删除已应用的stash
```

## 验证

```bash
git status  # 查看修改状态
```

