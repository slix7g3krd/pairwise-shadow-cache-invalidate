# pairwise-shadow-cache-invalidate

## 领域

游戏、图形与多媒体

## 主题

静态阴影缓存失效

## 说明

动态通道、全级联失效与页写入栅栏。

初始状态下 `graphics/shadow_cache_invalidate.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
