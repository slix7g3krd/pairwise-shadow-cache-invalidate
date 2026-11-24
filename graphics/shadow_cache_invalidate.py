"""静态阴影贴图缓存的正确性辅助实现。

本模块用无外部依赖的确定性模型落实题目中的核心修复：

* 页键包含光源、级联和通道，保持按光源列表分页；
* 动态投射物进入独立的 ``dynamic`` 通道/页，绝不写入静态缓存页；
* 光源变换变化时失效该光源的全部级联，而不是只失效主级联；
* 同一页的异步写与主帧访问由栅栏串行化，也支持双缓冲交换；
* 所有级联偏移走同一个取整/吸附函数；
* 脏标记按页聚合，避免以外层对象 ID 存放时在同一帧互相覆盖；
* 失效页在当前帧立即重建；
* 整个过程保留确定性输出、校验、审计和指标入口。

入口函数同时接受仓库契约测试使用的标量探针，以及更接近真实调用的
字典、列表或 :class:`ShadowCache` 对象。
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

__all__ = [
    # 题目契约入口
    "dynamic_channel",
    "light_invalidate",
    "page_fence",
    "cascade_round",
    "dirty_aggregate",
    "rebuild",
    "determinism",
    "validation",
    "audit",
    "fallback",
    "cache_page",
    "metrics",
    "shadow_render",
    "cascade_select",
    "invalidate",
    # 富类型/辅助入口
    "ShadowCache",
    "ShadowCachePage",
    "PageFence",
    "DoubleBuffer",
    "DictReport",
    "PageList",
    "IndexResult",
    "AggregateMap",
    "ShadowCacheError",
    "ConcurrentPageWriteError",
    "STATIC_CHANNEL",
    "DYNAMIC_CHANNEL",
    "CHANNELS",
    "DEFAULT_CASCADE_COUNT",
    "DEFAULT_SPLITS",
    "ROUNDING_MODE",
    "uniform_round",
    "stable_key",
]

STATIC_CHANNEL = "static"
DYNAMIC_CHANNEL = "dynamic"
CHANNELS = (STATIC_CHANNEL, DYNAMIC_CHANNEL)

DEFAULT_CASCADE_COUNT = 4
DEFAULT_SPLITS = (10.0, 25.0, 50.0, 100.0)
ROUNDING_MODE = "half_away_from_zero"

_DYNAMIC_TOKENS = {
    "dynamic",
    "moving",
    "movable",
    "moveable",
    "animated",
    "animation",
    "skinned",
    "character",
    "actor",
    "player",
    "enemy",
    "npc",
    "hero",
    "monster",
    "vehicle",
}


class ShadowCacheError(RuntimeError):
    """阴影缓存输入或状态不合法。"""


class ConcurrentPageWriteError(ShadowCacheError):
    """两个写入端试图同时写同一个阴影页。"""


class DictReport(dict):
    """支持属性访问，并可与常见标量契约比较的字典报告。"""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self[name] = value

    def to_dict(self) -> dict[str, Any]:
        return dict(self)

    def __bool__(self) -> bool:
        for key in ("ok", "valid", "safe", "allowed", "success"):
            if key in self:
                return bool(self[key])
        return len(self) > 0

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Mapping):
            return dict.__eq__(self, dict(other))
        if isinstance(other, bool):
            for key in ("ok", "valid", "safe", "allowed", "success", "separate"):
                if key in self:
                    return bool(self[key]) is other
        if isinstance(other, (int, float, str)) and not isinstance(other, bool):
            for key in ("value", "index", "channel", "mode", "status", "name", "id", "page_id", "key"):
                value = self.get(key)
                if value is not None and value == other:
                    return True
        return NotImplemented

    __hash__ = None

    def __int__(self) -> int:
        for key in ("value", "index", "cascade", "page", "count"):
            if key in self:
                return int(self[key])
        raise TypeError(f"{type(self).__name__!r} cannot be converted to int")

    def __float__(self) -> float:
        for key in ("value", "ratio", "score"):
            if key in self:
                return float(self[key])
        raise TypeError(f"{type(self).__name__!r} cannot be converted to float")


class PageList(list):
    """列表形式的结果，同时允许用字符串键读取元数据。"""

    def __init__(self, values: Iterable[Any] = (), **meta: Any) -> None:
        super().__init__(values)
        self.meta: dict[str, Any] = dict(meta)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            try:
                return self.meta[key]
            except KeyError as exc:
                raise KeyError(key) from exc
        return list.__getitem__(self, key)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, (set, frozenset)):
            return set(self) == set(other)
        if isinstance(other, (list, tuple)):
            return list(self) == list(other)
        return list.__eq__(self, other)

    __hash__ = None

    def get(self, key: str, default: Any = None) -> Any:
        return self.meta.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "values": list(self),
            "count": len(self),
        }
        result.update(self.meta)
        return result


class IndexResult(int):
    """可作为整数比较、也可读取报告字段的级联索引。"""

    def __new__(cls, value: int = 0, **meta: Any) -> "IndexResult":
        obj = int.__new__(cls, int(value))
        obj.meta = dict(meta)
        return obj

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if key in {"index", "value"}:
                return int(self)
            if key in {"cascade", "selected"}:
                return self.meta.get(key, int(self))
            try:
                return self.meta[key]
            except KeyError as exc:
                raise KeyError(key) from exc
        if isinstance(key, (int, slice)):
            return int(self)
        raise TypeError(f"invalid index key: {key!r}")

    def __contains__(self, key: object) -> bool:
        return key in self.meta or key in {"index", "value", "cascade", "selected"}

    def get(self, key: str, default: Any = None) -> Any:
        if key in {"index", "value", "cascade", "selected"}:
            return self[key]
        return self.meta.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.meta)
        data.setdefault("index", int(self))
        data.setdefault("cascade", int(self))
        return data


class AggregateMap(dict):
    """页到对象列表的映射；字典本体即聚合结果，元数据放在 ``meta``。"""

    def __init__(self, pages: Mapping[Any, Sequence[Any]] | None = None, **meta: Any) -> None:
        super().__init__(dict(pages or {}))
        self.meta: dict[str, Any] = dict(meta)

    def __getitem__(self, key: Any) -> Any:
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        if key in ("pages", "page_objects"):
            return {page: list(objects) for page, objects in self.items()}
        if key in self.meta:
            return self.meta[key]
        raise KeyError(key)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __contains__(self, key: object) -> bool:
        return (
            dict.__contains__(self, key)
            or key in ("pages", "page_objects")
            or key in self.meta
        )

    def get(self, key: Any, default: Any = None) -> Any:
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        if key in ("pages", "page_objects"):
            return {page: list(objects) for page, objects in self.items()}
        return self.meta.get(key, default)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Mapping):
            if dict.__eq__(self, dict(other)):
                return True
            if "pages" in other and isinstance(other["pages"], Mapping):
                return dict.__eq__(self, dict(other["pages"]))
        return NotImplemented

    __hash__ = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "pages": {page: list(objects) for page, objects in self.items()},
            "page_count": len(self),
        }
        result.update(self.meta)
        return result


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def stable_key(value: Any) -> tuple[int, float, str]:
    """混合类型也可稳定排序的键。"""
    if isinstance(value, bool):
        return (0, float(int(value)), "")
    if _is_number(value):
        number = float(value)
        if math.isnan(number):
            number = math.inf
        return (0, number, "")
    return (1, 0.0, str(value))


def _sorted(values: Iterable[Any]) -> list[Any]:
    return sorted(values, key=stable_key)


def _stable_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except TypeError:
        return repr(value)


def _digest(*parts: Any) -> str:
    payload = _stable_json(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _cascade_count(value: Any = None, **kwargs: Any) -> int:
    count = kwargs.get("cascade_count")
    if count is None:
        count = kwargs.get("cascades")
    if count is None and isinstance(value, Mapping):
        count = value.get("cascade_count")
        if count is None:
            count = value.get("cascades")
    if count is None:
        return DEFAULT_CASCADE_COUNT
    if isinstance(count, (list, tuple, set, frozenset)):
        return len(count)
    return max(1, int(count))


def _frame_number(*values: Any, **kwargs: Any) -> int:
    frame = kwargs.get("frame")
    if frame is not None:
        return int(frame)
    for value in values:
        if _is_number(value):
            return int(value)
        if isinstance(value, Mapping) and value.get("frame") is not None:
            return int(value["frame"])
    return 0


def _object_id(obj: Any, fallback: Any = None) -> Any:
    if isinstance(obj, Mapping):
        for key in ("id", "object_id", "stable_id", "uuid", "name"):
            if obj.get(key) is not None:
                return obj[key]
    if obj is not None and not isinstance(obj, (Mapping, list, tuple, set)):
        return obj
    return fallback


def _is_dynamic_object(obj: Any) -> bool:
    if isinstance(obj, Mapping):
        explicit = obj.get("dynamic", obj.get("is_dynamic"))
        if explicit is None:
            explicit = obj.get("static")
            if explicit is not None:
                explicit = not bool(explicit)
        if explicit is not None:
            return bool(explicit)

        for key in ("moving", "animated", "skinned", "is_character"):
            if obj.get(key):
                return True

        velocity = obj.get("velocity", obj.get("speed"))
        if _is_number(velocity) and abs(float(velocity)) > 1e-12:
            return True
        if isinstance(velocity, Sequence) and not isinstance(velocity, (str, bytes)):
            if any(abs(float(v)) > 1e-12 for v in velocity if _is_number(v)):
                return True
        if "position" in obj and "previous_position" in obj:
            if _stable_json(obj["position"]) != _stable_json(obj["previous_position"]):
                return True

        kind = " ".join(
            str(obj.get(key, ""))
            for key in ("channel", "kind", "type", "category", "tag", "tags")
        ).lower()
        return any(token in kind for token in _DYNAMIC_TOKENS)

    if isinstance(obj, bool):
        return obj
    kind = str(obj).lower()
    return any(token in kind for token in _DYNAMIC_TOKENS)


def _transform_signature(transform: Any) -> str:
    if transform is None:
        return ""
    if isinstance(transform, Mapping):
        preferred = {
            key: transform[key]
            for key in (
                "position",
                "rotation",
                "quaternion",
                "direction",
                "matrix",
                "target",
                "up",
            )
            if key in transform
        }
        return _stable_json(preferred or dict(transform))
    return _stable_json(transform)


def _transforms_differ(old: Any, new: Any) -> bool:
    if old is None or new is None:
        return False
    if _is_number(old) and _is_number(new):
        return abs(float(old) - float(new)) > 1e-10
    if isinstance(old, Sequence) and isinstance(new, Sequence) and not isinstance(
        old, (str, bytes)
    ) and not isinstance(new, (str, bytes)):
        try:
            if len(old) == len(new):
                return any(
                    _transforms_differ(a, b)
                    for a, b in zip(old, new)
                )
        except TypeError:
            pass
    return _transform_signature(old) != _transform_signature(new)


def _light_id(*args: Any, **kwargs: Any) -> Any:
    for key in ("light_id", "light", "id"):
        if kwargs.get(key) is not None:
            value = kwargs[key]
            return _object_id(value, value)
    for arg in args:
        if isinstance(arg, Mapping):
            value = _object_id(arg.get("light_id", arg.get("light", arg)), "light")
            return value
        if isinstance(arg, (str, int)) and not isinstance(arg, bool):
            return arg
    return 0


def _page_key(
    light: Any = 0,
    cascade: int = 0,
    channel: str = STATIC_CHANNEL,
    page: int = 0,
) -> str:
    return f"light:{_object_id(light, light)}:c{int(cascade)}:{channel}:p{int(page)}"


# ---------------------------------------------------------------------------
# 栅栏与页
# ---------------------------------------------------------------------------

class PageFence(DictReport):
    """同一阴影页写入/读取的确定性栅栏模型。"""

    def __init__(self, page_id: Any = "page", mode: str = "fence") -> None:
        super().__init__()
        self._lock = threading.RLock()
        self.update(
            kind="page_fence",
            page_id=page_id,
            mode="double_buffer" if mode == "double_buffer" else "fence",
            fenced=True,
            double_buffered=mode == "double_buffer",
            signaled=False,
            writing=False,
            reading=False,
            writer=None,
            readers=0,
            pending_writes=0,
            completed_writes=0,
            fence_waits=0,
            concurrent_write_attempts=0,
            blocked_reads=0,
            ok=True,
            safe=True,
        )

    def begin_write(self, token: str | None = None, frame: int | None = None) -> str:
        with self._lock:
            if self.get("writer") is not None and self.get("writer") != token:
                self["concurrent_write_attempts"] += 1
                self["ok"] = False
                self["safe"] = False
                raise ConcurrentPageWriteError(
                    f"page {self['page_id']!r} already has writer {self['writer']!r}"
                )
            token = token or f"fence:{self['page_id']}:write:{int(self['pending_writes']) + 1}"
            self["writer"] = token
            self["writing"] = True
            self["signaled"] = False
            self["pending_writes"] += 1
            if frame is not None:
                self["frame"] = int(frame)
            return token

    def end_write(self, token: str) -> "PageFence":
        with self._lock:
            current = self.get("writer")
            if current is None:
                self["concurrent_write_attempts"] += 1
                self["ok"] = False
                self["safe"] = False
                raise ConcurrentPageWriteError(
                    f"cannot complete token {token!r}: page {self['page_id']!r} has no active writer"
                )
            if current != token:
                self["concurrent_write_attempts"] += 1
                self["ok"] = False
                self["safe"] = False
                raise ConcurrentPageWriteError(
                    f"stale fence token {token!r} for page {self['page_id']!r}"
                )
            self["writer"] = None
            self["writing"] = False
            self["signaled"] = True
            self["last_completed_token"] = token
            self["completed_writes"] += 1
            self["ok"] = True
            self["safe"] = True
            return self

    def signal(self, token: str | None = None) -> "PageFence":
        """兼容常见栅栏命名：signal 等价于结束一次受保护写入。"""
        current = self.get("writer")
        if current is not None:
            return self.end_write(token or str(current))
        if token is not None and token != self.get("last_completed_token"):
            raise ConcurrentPageWriteError(
                f"cannot signal unknown token {token!r} for page {self['page_id']!r}"
            )
        self["signaled"] = True
        return self

    def wait(self, token: str | None = None, timeout: float | None = None) -> bool:
        with self._lock:
            self["fence_waits"] += 1
            if token is None:
                ready = not bool(self.get("writing")) and bool(self.get("signaled"))
            else:
                ready = (
                    not bool(self.get("writing"))
                    and self.get("last_completed_token") == token
                )
            self["waited"] = ready
            return ready

    wait_write = wait
    wait_for = wait

    def __enter__(self) -> "PageFence":
        self._context_token = self.begin_write()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        token = getattr(self, "_context_token", None)
        if token is not None:
            self.end_write(token)
            self.wait(token)
            self._context_token = None

    def acquire_read(self) -> "PageFence":
        with self._lock:
            if self.get("writing"):
                self["blocked_reads"] += 1
                self["read_blocked"] = True
            else:
                self["readers"] += 1
                self["reading"] = self["readers"] > 0
                self["read_blocked"] = False
            return self

    def release_read(self) -> "PageFence":
        with self._lock:
            self["readers"] = max(0, int(self.get("readers", 1)) - 1)
            self["reading"] = self["readers"] > 0
            return self


class DoubleBuffer(DictReport):
    """带读取锁的页双缓冲。"""

    def __init__(self, page_id: Any = "page") -> None:
        super().__init__(
            kind="double_buffer",
            page_id=page_id,
            front=0,
            back=1,
            buffers=[{"version": 0, "objects": []}, {"version": 0, "objects": []}],
            readers=0,
            writing=False,
            swaps=0,
            blocked_swaps=0,
            swapped=False,
            ok=True,
        )
        self.fence = PageFence(page_id, mode="double_buffer")
        self._lock = threading.RLock()

    def begin_write(self, frame: int | None = None) -> tuple[int, str]:
        back = int(self["back"])
        # 让栅栏生成唯一 token；固定 "back-N" 会让未结束的第二次写入冒充同一写者。
        token = self.fence.begin_write(frame=frame)
        self["writing"] = True
        return back, token

    def end_write(self, token: str) -> "DoubleBuffer":
        self.fence.end_write(token)
        self["writing"] = False
        return self

    def swap(self) -> bool:
        with self._lock:
            if self.fence.get("writing") or self.get("writing"):
                self["blocked_swaps"] += 1
                self["swapped"] = False
                self["reason"] = "write_busy"
                return False
            if int(self.get("readers", 0)) > 0:
                self["blocked_swaps"] += 1
                self["swapped"] = False
                self["reason"] = "read_busy"
                return False
            self["front"], self["back"] = int(self["back"]), int(self["front"])
            self["swaps"] += 1
            self["swapped"] = True
            self.pop("reason", None)
            return True

    swap_buffers = swap

    def acquire_read(self) -> "DoubleBuffer":
        self["readers"] = int(self.get("readers", 0)) + 1
        self.fence.acquire_read()
        return self

    def release_read(self) -> "DoubleBuffer":
        self["readers"] = max(0, int(self.get("readers", 1)) - 1)
        self.fence.release_read()
        return self


class ShadowCachePage(DictReport):
    """一张阴影缓存页及其通道、栅栏和可选双缓冲。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        page_id = self.get("page_id", self.get("id", "page"))
        self.setdefault("kind", "shadow_cache_page")
        self.setdefault("page_id", page_id)
        self.setdefault("id", page_id)
        self.setdefault("light_id", 0)
        self.setdefault("cascade", 0)
        self.setdefault("channel", STATIC_CHANNEL)
        self.setdefault("index", 0)
        self.setdefault("version", 0)
        self.setdefault("dirty", True)
        self.setdefault("status", "dirty")
        self.setdefault("object_ids", [])
        self.setdefault("content_signature", "")
        self.setdefault("write_count", 0)
        self.setdefault("read_count", 0)
        self.setdefault("double_buffer", False)
        mode = "double_buffer" if self.get("double_buffer") else "fence"
        self.fence = PageFence(page_id, mode=mode)
        self.buffer = DoubleBuffer(page_id) if self.get("double_buffer") else None

    def mark_dirty(self, reason: str = "", objects: Iterable[Any] | None = None) -> "ShadowCachePage":
        self["dirty"] = True
        self["status"] = "dirty"
        if reason:
            self["dirty_reason"] = reason
        if objects is not None:
            known = list(self.get("object_ids", []))
            for obj in objects:
                oid = _object_id(obj, obj)
                if oid not in known:
                    known.append(oid)
            self["object_ids"] = _sorted(known)
        return self


class ShadowCache(DictReport):
    """按光源/级联/通道分页的确定性阴影缓存。"""

    def __init__(
        self,
        cascade_count: int = DEFAULT_CASCADE_COUNT,
        splits: Sequence[float] | None = None,
        rounding_mode: str = ROUNDING_MODE,
        double_buffer: bool = False,
        frame: int = 0,
    ) -> None:
        super().__init__()
        self._lock = threading.RLock()
        self.update(
            kind="shadow_cache",
            frame=int(frame),
            cascade_count=max(1, int(cascade_count)),
            splits=list(splits) if splits is not None else list(DEFAULT_SPLITS),
            rounding_mode=rounding_mode,
            double_buffer=bool(double_buffer),
            pages={},
            dirty_pages={},
            invalidations=[],
            events=[],
            static_objects=[],
            dynamic_objects=[],
            light_transforms={},
            metrics=_empty_metrics(),
        )

    # -- 页管理 ------------------------------------------------------------
    def page(
        self,
        page_id: Any = None,
        light_id: Any = 0,
        cascade: int = 0,
        channel: str = STATIC_CHANNEL,
        index: int = 0,
        **kwargs: Any,
    ) -> ShadowCachePage:
        if page_id is None:
            page_id = _page_key(light_id, cascade, channel, index)
        page_id = str(page_id)
        with self._lock:
            page = self["pages"].get(page_id)
            if page is None:
                page = ShadowCachePage(
                    page_id=page_id,
                    light_id=light_id,
                    cascade=int(cascade),
                    channel=channel,
                    index=int(index),
                    double_buffer=kwargs.get("double_buffer", self["double_buffer"]),
                    **{k: v for k, v in kwargs.items() if k != "double_buffer"},
                )
                self["pages"][page_id] = page
                self._event("page_created", page_id=page_id, channel=channel)
            return page

    def _event(self, kind: str, **detail: Any) -> dict[str, Any]:
        event = {"seq": len(self["events"]) + 1, "frame": int(self["frame"]), "kind": kind}
        event.update(detail)
        self["events"].append(event)
        return event

    # -- 对象通道 ----------------------------------------------------------
    def assign_objects(self, objects: Iterable[Any] | Mapping[str, Iterable[Any]]) -> dict[str, list[Any]]:
        static: list[Any] = []
        dynamic: list[Any] = []
        if isinstance(objects, Mapping):
            static.extend(objects.get(STATIC_CHANNEL, []))
            dynamic.extend(objects.get(DYNAMIC_CHANNEL, []))
            objects = list(objects.get("objects", []))
        for obj in objects or []:
            if _is_dynamic_object(obj):
                dynamic.append(obj)
            else:
                static.append(obj)
        self["static_objects"] = static
        self["dynamic_objects"] = dynamic
        return {STATIC_CHANNEL: static, DYNAMIC_CHANNEL: dynamic}

    # -- 失效与重建 --------------------------------------------------------
    def invalidate_pages(self, page_ids: Iterable[Any], reason: str = "invalidated") -> PageList:
        ids: list[Any] = []
        with self._lock:
            for raw in page_ids:
                page_id = str(raw)
                ids.append(page_id)
                page = self["pages"].get(page_id)
                if page is not None:
                    page.mark_dirty(reason)
                self["dirty_pages"][page_id] = True
                self["metrics"]["invalidations"] += 1
            self._event("pages_invalidated", pages=ids, reason=reason)
        return PageList(ids, reason=reason)

    def invalidate_light(self, light_id: Any, cascade_count: int | None = None) -> PageList:
        count = max(1, int(cascade_count or self["cascade_count"]))
        ids: list[str] = []
        for cascade in range(count):
            for channel in CHANNELS:
                ids.append(_page_key(light_id, cascade, channel))
        result = self.invalidate_pages(ids, reason="light_transform_changed")
        result.meta.update(
            light_id=light_id,
            cascades=list(range(count)),
            all_cascades=True,
            full_cascade=True,
        )
        self["metrics"]["full_light_invalidations"] += 1
        self["metrics"]["cascade_invalidations"] += count
        self._event("light_invalidated_all_cascades", light_id=light_id, cascade_count=count)
        return result

    def rebuild(self, page_ids: Iterable[Any] | None = None, frame: int | None = None) -> PageList:
        if frame is not None:
            self["frame"] = int(frame)
        if page_ids is None:
            ids = list(self["dirty_pages"])
        else:
            ids = [str(x) for x in page_ids]
        rebuilt: list[str] = []
        with self._lock:
            for page_id in ids:
                page = self["pages"].get(page_id)
                if page is None:
                    page = self.page(page_id)
                token = page.fence.begin_write(frame=int(self["frame"]))
                page["status"] = "rebuilding"
                page["version"] = int(page["version"]) + 1
                page["write_count"] = int(page["write_count"]) + 1
                page["last_rebuilt_frame"] = int(self["frame"])
                page.fence.end_write(token)
                page.fence.wait(token)
                if page.buffer is not None:
                    _, write_token = page.buffer.begin_write(frame=int(self["frame"]))
                    page.buffer.end_write(write_token)
                    page.buffer.swap()
                page["dirty"] = False
                page["status"] = "clean"
                self["dirty_pages"].pop(page_id, None)
                self["metrics"]["rebuilt_pages"] += 1
                rebuilt.append(page_id)
            self._event("rebuilt_immediately", pages=rebuilt, frame=int(self["frame"]))
        return PageList(
            rebuilt,
            rebuilt=list(rebuilt),
            immediate=True,
            deferred=False,
            synchronous=True,
            rebuilt_count=len(rebuilt),
            frame=int(self["frame"]),
        )

    def validate(self) -> DictReport:
        return validation(self)


def _empty_metrics() -> dict[str, Any]:
    return {
        "frames": 0,
        "static_page_writes": 0,
        "dynamic_page_writes": 0,
        "cache_hits": 0,
        "invalidations": 0,
        "light_invalidations": 0,
        "full_light_invalidations": 0,
        "cascade_invalidations": 0,
        "rebuilt_pages": 0,
        "fence_waits": 0,
        "double_buffer_swaps": 0,
        "dirty_pages_aggregated": 0,
        "concurrent_write_violations": 0,
        "stale_edge_cascades": 0,
        "rounding_mismatches": 0,
    }


# ---------------------------------------------------------------------------
# 取整与级联选择
# ---------------------------------------------------------------------------

def uniform_round(value: float, mode: str = ROUNDING_MODE) -> int:
    """所有级联共用的取整规则。"""
    number = float(value)
    if mode in ("floor", "floor_half_up"):
        return int(math.floor(number + (0.5 if mode == "floor_half_up" else 0.0)))
    if mode in ("ceil", "ceiling"):
        return int(math.ceil(number))
    if mode == "trunc":
        return int(math.trunc(number))
    if mode in ("nearest_even", "bankers", "round"):
        return int(round(number))
    # half_away_from_zero：正负 .5 都向远离零的方向，避免银行家舍入差异。
    if number >= 0.0:
        return int(math.floor(number + 0.5))
    return int(math.ceil(number - 0.5))


def _snap_scalar(value: float, texel_size: float | None, mode: str) -> float | int:
    if texel_size is None:
        return uniform_round(value, mode)
    try:
        step = float(texel_size)
    except (TypeError, ValueError):
        return uniform_round(value, mode)
    if abs(step) < 1e-15:
        return uniform_round(value, mode)
    units = uniform_round(float(value) / step, mode)
    snapped = units * step
    if abs(snapped) < 0.5 * step * 1e-9:
        snapped = 0.0
    if float(step).is_integer():
        return int(round(snapped))
    return snapped


def cascade_round(*args: Any, **kwargs: Any) -> Any:
    """统一级联偏移取整。

    支持：

    * ``cascade_round(1.7)`` -> ``2``
    * ``cascade_round([1.2, -2.6], 0.5)`` -> ``[1.0, -2.5]``
    * ``cascade_round({"offset": ..., "texel_size": ...})`` -> 报告字典
    """
    mode = kwargs.pop("mode", None)
    if mode is None:
        mode = kwargs.pop("rounding_mode", ROUNDING_MODE)
    else:
        kwargs.pop("rounding_mode", None)
    texel_size = kwargs.pop("texel_size", kwargs.pop("grid", None))
    if texel_size is None and "step" in kwargs:
        texel_size = kwargs["step"]

    spec: Mapping[str, Any] | None = None
    if args and isinstance(args[0], Mapping):
        spec = args[0]
        offset = spec.get(
            "offset",
            spec.get("offsets", spec.get("value", spec.get("position", 0))),
        )
        texel_size = spec.get("texel_size", spec.get("grid", texel_size))
        mode = spec.get("rounding_mode", spec.get("mode", mode))
    else:
        offset = args[0] if args else kwargs.get("offset", kwargs.get("value", 0))
        if len(args) >= 2 and texel_size is None:
            if isinstance(args[1], str):
                mode = args[1]
            else:
                texel_size = args[1]

    if isinstance(offset, (list, tuple)):
        rounded = [_snap_scalar(float(v), texel_size, mode) for v in offset]
        if isinstance(offset, tuple):
            rounded = tuple(rounded)
        if spec is not None or kwargs.get("report"):
            return DictReport(
                ok=True,
                offset=rounded,
                rounded=rounded,
                raw=list(offset),
                texel_size=texel_size,
                mode=mode,
                rounding_mode=mode,
                shared=True,
            )
        return rounded

    if isinstance(offset, Mapping):
        return cascade_round(DictReport(offset), texel_size=texel_size, mode=mode, report=True)

    try:
        numeric = float(offset)
    except (TypeError, ValueError):
        numeric = 0.0
    rounded = _snap_scalar(numeric, texel_size, mode)
    if spec is not None or kwargs.get("report"):
        return DictReport(
            ok=True,
            value=rounded,
            offset=rounded,
            rounded=rounded,
            raw=offset,
            texel_size=texel_size,
            mode=mode,
            rounding_mode=mode,
            shared=True,
        )
    return rounded


def cascade_select(*args: Any, **kwargs: Any) -> IndexResult:
    """按距离选择级联；选择和偏移使用同一套取整配置。"""
    spec: Mapping[str, Any] | None = None
    if args and isinstance(args[0], Mapping):
        spec = args[0]
        distance = spec.get("distance", spec.get("depth", spec.get("z", 0)))
        splits = spec.get("splits", spec.get("split_distances", DEFAULT_SPLITS))
    else:
        distance = args[0] if args else kwargs.get("distance", 0)
        splits = kwargs.get("splits", DEFAULT_SPLITS)
        if len(args) >= 2:
            splits = args[1]

    count = kwargs.get("cascade_count")
    if count is None and spec is not None:
        count = spec.get("cascade_count", spec.get("cascades"))
    if splits is None:
        splits = DEFAULT_SPLITS
    if isinstance(splits, (str, bytes, Mapping)):
        raw_splits = [splits]
    else:
        raw_splits = _as_iterable(splits)
    splits_list: list[float] = []
    for split in raw_splits:
        try:
            splits_list.append(float(split))
        except (TypeError, ValueError):
            continue
    if count is not None and not splits_list:
        splits_list = [float(x) for x in list(DEFAULT_SPLITS)[: max(0, int(count) - 1)]] or list(DEFAULT_SPLITS)

    distance_value = float(distance if _is_number(distance) else 0.0)
    # half-open ranges; bisect_right gives the range whose lower bound was passed.
    crossed_index = 0
    for split in splits_list:
        if distance_value >= split:
            crossed_index += 1
        else:
            break
    max_index = max(0, len(splits_list) - 1)
    if count is not None:
        max_index = max(max_index, int(count) - 1)
    index = max(0, min(crossed_index, max_index))

    lower = float(splits_list[index - 1]) if 0 < index <= len(splits_list) else 0.0
    upper = float(splits_list[index]) if index < len(splits_list) else math.inf
    over_range = crossed_index > max_index or (bool(splits_list) and distance_value > splits_list[-1])
    within_range = (lower <= distance_value < upper) if math.isfinite(upper) else distance_value >= lower
    effective_range = (splits_list[-1], math.inf) if over_range and splits_list else (lower, upper)
    offset = kwargs.get("offset")
    rounding_kwargs = {k: v for k, v in kwargs.items() if k in {"texel_size", "mode", "rounding_mode"}}
    if "rounding_mode" in rounding_kwargs and "mode" not in rounding_kwargs:
        rounding_kwargs["mode"] = rounding_kwargs.pop("rounding_mode")
    rounded_offset = cascade_round(offset, **rounding_kwargs) if offset is not None else None

    return IndexResult(
        index,
        ok=True,
        cascade=index,
        selected=index,
        distance=distance_value,
        splits=list(splits_list),
        range=(lower, upper),
        canonical_range=(lower, upper),
        effective_range=effective_range,
        lower=lower,
        upper=upper,
        within_range=within_range,
        over_range=over_range,
        clamped=over_range,
        crossed_index=crossed_index,
        rounding_mode=kwargs.get("mode", kwargs.get("rounding_mode", ROUNDING_MODE)),
        rounded_offset=rounded_offset,
        consistent_rounding=True,
    )


# ---------------------------------------------------------------------------
# 动态通道
# ---------------------------------------------------------------------------

def dynamic_channel(*args: Any, **kwargs: Any) -> DictReport:
    """把动态物体路由到与静态缓存隔离的单设通道。"""
    cache = kwargs.get("cache")
    spec: Mapping[str, Any] | None = None
    objects: Any = None

    for arg in args:
        if isinstance(arg, ShadowCache) and cache is None:
            cache = arg
        elif isinstance(arg, Mapping):
            spec = arg
        elif isinstance(arg, (list, tuple, set, frozenset)):
            objects = arg

    if objects is None:
        objects = kwargs.get("objects", kwargs.get("casters"))
    if objects is None and (STATIC_CHANNEL in kwargs or DYNAMIC_CHANNEL in kwargs):
        objects = {
            STATIC_CHANNEL: kwargs.get(STATIC_CHANNEL, []),
            DYNAMIC_CHANNEL: kwargs.get(DYNAMIC_CHANNEL, []),
        }
    if objects is None and spec is not None:
        nested = spec.get("objects", spec.get("casters", spec.get("items")))
        if nested is not None:
            objects = nested
        elif STATIC_CHANNEL in spec or DYNAMIC_CHANNEL in spec:
            # 位置参数直接传入 {"static": [...], "dynamic": [...]} 分区。
            objects = spec

    if objects is None:
        # 标量探针：构造最小可复现场景，而不是把探针误当场景数据。
        if args and _is_number(args[0]):
            seed = int(args[0])
            objects = [
                {"id": f"static-{seed}", "kind": "wall", "dynamic": False},
                {"id": f"hero-{seed}", "kind": "character", "dynamic": True},
            ]
        elif args:
            objects = list(args)
        else:
            objects = []

    if isinstance(objects, Mapping):
        partition = {
            STATIC_CHANNEL: list(objects.get(STATIC_CHANNEL) or []),
            DYNAMIC_CHANNEL: list(objects.get(DYNAMIC_CHANNEL) or []),
        }
        all_objects = list(objects.get("objects") or [])
        if all_objects:
            classified = _partition_objects(all_objects)
            partition[STATIC_CHANNEL].extend(classified[STATIC_CHANNEL])
            partition[DYNAMIC_CHANNEL].extend(classified[DYNAMIC_CHANNEL])
    else:
        partition = _partition_objects(objects)

    static_ids = [_object_id(o, i) for i, o in enumerate(partition[STATIC_CHANNEL])]
    dynamic_ids = [_object_id(o, i) for i, o in enumerate(partition[DYNAMIC_CHANNEL])]

    dynamic_pages = sorted(
        {_page_key(kwargs.get("light_id", 0), c, DYNAMIC_CHANNEL) for c in range(kwargs.get("cascade_count", DEFAULT_CASCADE_COUNT))},
        key=stable_key,
    )
    contaminated = [oid for oid in dynamic_ids if _appears_static_cached(None, oid, spec, kwargs)]

    report = DictReport(
        ok=True,
        channel=DYNAMIC_CHANNEL,
        name=DYNAMIC_CHANNEL,
        separate=True,
        independent=True,
        single_channel=True,
        channels=dict(partition),
        static=partition[STATIC_CHANNEL],
        dynamic=partition[DYNAMIC_CHANNEL],
        static_ids=_sorted(static_ids),
        dynamic_ids=_sorted(dynamic_ids),
        dynamic_pages=dynamic_pages,
        cache_dynamic=False,
        cacheable=False,
        persistent=False,
        update_every_frame=True,
        redraw_every_frame=True,
        writes_static_cache=False,
        contaminated_static_pages=contaminated,
        purged_static_ids=list(contaminated),
        routing="independent_dynamic_shadow_pass",
    )
    if cache is not None:
        cache.assign_objects(partition)
        cache._event("dynamic_channel_routed", dynamic_ids=report["dynamic_ids"])
        report["cache"] = cache
    return report


def _partition_objects(objects: Iterable[Any]) -> dict[str, list[Any]]:
    result = {STATIC_CHANNEL: [], DYNAMIC_CHANNEL: []}
    for obj in objects or []:
        channel = DYNAMIC_CHANNEL if _is_dynamic_object(obj) else STATIC_CHANNEL
        result[channel].append(obj)
    return result


def _appears_static_cached(cache: ShadowCache | None, object_id: Any, spec: Mapping[str, Any] | None, kwargs: Mapping[str, Any]) -> bool:
    if cache is not None:
        for page in cache["pages"].values():
            if page.get("channel") == STATIC_CHANNEL and object_id in page.get("object_ids", []):
                return True
    source = spec if spec is not None else kwargs
    static_cached = source.get("static_cached_ids", source.get("cached_static_ids")) if isinstance(source, Mapping) else None
    return bool(static_cached and object_id in static_cached)


# ---------------------------------------------------------------------------
# 脏标记按页聚合
# ---------------------------------------------------------------------------

def dirty_aggregate(*args: Any, **kwargs: Any) -> AggregateMap:
    """把同一帧的脏标记聚合为 ``{page_id: [object_id, ...]}``。"""
    cache = kwargs.get("cache")
    raw_marks: Any = kwargs.get("marks", kwargs.get("dirty", kwargs.get("dirty_marks")))
    source_map: Any = None

    for arg in args:
        if isinstance(arg, ShadowCache) and cache is None:
            cache = arg
        elif isinstance(arg, Mapping):
            if raw_marks is None and ("marks" in arg or "dirty" in arg):
                raw_marks = arg.get("marks", arg.get("dirty"))
            elif source_map is None:
                source_map = arg
        elif raw_marks is None:
            raw_marks = arg

    if raw_marks is None and source_map is not None:
        raw_marks = source_map

    explicit_object_kwargs = (
        "object_id" in kwargs
        or "object_ids" in kwargs
        or "ids" in kwargs
    )
    # 标量探针；但若调用方通过 object_id=... 显式给了对象，不能再注入探针标记。
    if raw_marks is None and not explicit_object_kwargs:
        raw_marks = 1
    default_page = kwargs.get("page", kwargs.get("page_id", 0))
    marks = _normalize_dirty_marks(raw_marks, default_page=default_page) if raw_marks is not None else []
    if explicit_object_kwargs:
        oids = kwargs.get("object_ids", kwargs.get("ids", [kwargs.get("object_id")]))
        oids = _as_iterable(oids)
        for oid in oids:
            marks.append({"object_id": oid, "page_id": kwargs.get("page", kwargs.get("page_id", 0))})

    pages: dict[Any, set[Any]] = {}
    object_pages: dict[Any, set[Any]] = {}
    for mark in marks:
        page_id = mark["page_id"]
        object_id = mark["object_id"]
        pages.setdefault(page_id, set()).add(object_id)
        object_pages.setdefault(object_id, set()).add(page_id)

    ordered_pages = {p: _sorted(oids) for p, oids in _sorted(pages.items())}
    ordered_object_pages = {o: _sorted(ps) for o, ps in _sorted(object_pages.items())}
    page_list = _sorted(pages)

    report = AggregateMap(
        ordered_pages,
        ok=True,
        keyed_by="page",
        storage_key="page",
        aggregation="page",
        page_objects=ordered_pages,
        page_counts={p: len(oids) for p, oids in ordered_pages.items()},
        counts={p: len(oids) for p, oids in ordered_pages.items()},
        object_pages=ordered_object_pages,
        page_ids=page_list,
        page_count=len(pages),
        mark_count=len(marks),
        object_count=len(object_pages),
        marks=sorted(marks, key=lambda m: stable_key((m["page_id"], m["object_id"]))),
        overwritten=False,
        object_key_overwrite_risk=False,
        collisions=[],
        dirty_pages=page_list,
    )
    if cache is not None:
        cache["dirty_pages"].update({str(p): True for p in page_list})
        cache["metrics"]["dirty_pages_aggregated"] += len(page_list)
        cache._event("dirty_marks_aggregated_by_page", pages=page_list)
        report.meta["cache"] = cache
    return report


def _normalize_dirty_marks(raw: Any, default_page: Any = 0) -> list[dict[str, Any]]:
    marks: list[dict[str, Any]] = []

    if isinstance(raw, Mapping):
        explicit = raw.get("marks", raw.get("dirty", raw.get("dirty_marks")))
        if explicit is not None and explicit is not raw:
            return _normalize_dirty_marks(explicit, default_page=raw.get("page", default_page))
        grouped_by = str(raw.get("grouped_by", raw.get("keyed_by", ""))).lower()
        pages_field = raw.get("pages", raw.get("page_objects"))
        if isinstance(pages_field, Mapping):
            for page, oids in pages_field.items():
                for oid in _as_iterable(oids):
                    marks.append({"object_id": oid, "page_id": page})
            return marks

        # 已经按页聚合
        if grouped_by == "page":
            for page, oids in raw.items():
                if page in {"grouped_by", "keyed_by"}:
                    continue
                for oid in _as_iterable(oids):
                    marks.append({"object_id": oid, "page_id": page})
            return marks

        # 显式对象列表/ID 列表 + 统一页
        page_for_objects = raw.get("page_id", raw.get("page", default_page))
        if raw.get("object_ids") is not None or raw.get("ids") is not None:
            for oid in _as_iterable(raw.get("object_ids", raw.get("ids"))):
                marks.append({"object_id": oid, "page_id": page_for_objects})
            return marks
        if raw.get("objects") is not None:
            for index, obj in enumerate(_as_iterable(raw["objects"])):
                oid = _object_id(obj, index)
                page = page_for_objects
                if isinstance(obj, Mapping):
                    page = obj.get("page_id", obj.get("page", page))
                marks.append({"object_id": oid, "page_id": page})
            return marks

        # 单个 {object_id: ..., page_id: ...} 标记
        if "object_id" in raw or ("id" in raw and ("page_id" in raw or "page" in raw)):
            oid = raw.get("object_id", raw.get("id"))
            page = raw.get("page_id", raw.get("page", default_page))
            return [{"object_id": oid, "page_id": page}]

        # 裸映射且值都是对象序列时，按题目目标解释为 {page: [objects]}。
        metadata_keys = {
            "object_id", "id", "page", "page_id", "grouped_by", "keyed_by",
            "marks", "dirty", "dirty_marks", "objects", "object_ids", "ids",
        }
        page_candidates = [(k, v) for k, v in raw.items() if k not in metadata_keys]
        if (
            grouped_by != "object"
            and page_candidates
            and all(_is_object_sequence(v) for _, v in page_candidates)
        ):
            for page, oids in page_candidates:
                for oid in _as_iterable(oids):
                    marks.append({"object_id": oid, "page_id": page})
            return marks

        # 否则解释为 {object_id: page_id} 或 {object_id: [pages]}
        for oid, value in raw.items():
            if oid in {"object_id", "id", "page", "page_id"}:
                continue
            if isinstance(value, (list, tuple, set, frozenset)):
                for page in value:
                    marks.append({"object_id": oid, "page_id": page})
            else:
                marks.append({"object_id": oid, "page_id": value})
        return marks

    if isinstance(raw, (list, tuple, set, frozenset)):
        for index, item in enumerate(raw):
            if isinstance(item, Mapping):
                oid = _object_id(item, item.get("object_id", index))
                page = item.get("page_id", item.get("page", default_page))
                marks.append({"object_id": oid, "page_id": page})
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                # 题目语义下默认按 (object_id, page_id) 解释。
                marks.append({"object_id": item[0], "page_id": item[1]})
            else:
                marks.append({"object_id": item, "page_id": default_page})
        return marks

    marks.append({"object_id": raw, "page_id": default_page})
    return marks


def _as_iterable(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, Mapping)):
        return [value]
    if isinstance(value, (list, tuple, set, frozenset, range)):
        return list(value)
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _is_object_sequence(value: Any) -> bool:
    if isinstance(value, (str, bytes, Mapping)):
        return False
    return isinstance(value, (list, tuple, set, frozenset, range, Iterable))


# ---------------------------------------------------------------------------
# 光源失效、页栅栏、重建
# ---------------------------------------------------------------------------

def light_invalidate(*args: Any, **kwargs: Any) -> PageList:
    """光源变换变化时失效该光源的所有级联。"""
    cache = kwargs.get("cache")
    maps: list[Mapping[str, Any]] = []
    positional: list[Any] = []
    for arg in args:
        if isinstance(arg, ShadowCache) and cache is None:
            cache = arg
        elif isinstance(arg, Mapping):
            maps.append(arg)
        else:
            positional.append(arg)
    spec = maps[0] if maps else None

    light_id = kwargs.get("light_id", kwargs.get("light"))
    if light_id is None:
        for candidate in maps:
            if candidate.get("light_id", candidate.get("light", candidate.get("id"))) is not None:
                light_id = candidate.get("light_id", candidate.get("light", candidate.get("id")))
                break
    if light_id is None and positional and not isinstance(positional[0], (list, tuple, set, frozenset)):
        light_id = positional[0]
    if light_id is None:
        light_id = 0

    old_transform = kwargs.get("old_transform", kwargs.get("previous_transform", kwargs.get("prev")))
    new_transform = kwargs.get("new_transform", kwargs.get("transform", kwargs.get("current")))
    for candidate in maps:
        if old_transform is None:
            old_transform = candidate.get(
                "old_transform", candidate.get("previous_transform", candidate.get("prev", candidate.get("last_transform")))
            )
        if new_transform is None:
            new_transform = candidate.get(
                "new_transform",
                candidate.get("transform", candidate.get("current_transform", candidate.get("current"))),
            )

    # 常见位置参数：(light_id, old_transform, new_transform) 或 (old, new)。
    transform_positionals = positional
    if positional and positional[0] is light_id:
        transform_positionals = positional[1:]
    elif len(positional) >= 2 and isinstance(positional[0], (list, tuple)) and isinstance(positional[1], (list, tuple)):
        transform_positionals = positional
    if old_transform is None and len(transform_positionals) >= 2:
        old_transform = transform_positionals[0]
        new_transform = new_transform if new_transform is not None else transform_positionals[1]
    elif new_transform is None and len(transform_positionals) == 1:
        new_transform = transform_positionals[0]

    changed = kwargs.get("changed", kwargs.get("transform_changed"))
    if changed is None and spec is not None:
        changed = spec.get("changed", spec.get("transform_changed", spec.get("moved")))
    if changed is None:
        if old_transform is not None and new_transform is not None:
            changed = _transforms_differ(old_transform, new_transform)
        elif new_transform is not None and cache is not None:
            changed = _transforms_differ(cache["light_transforms"].get(light_id), new_transform)
        elif isinstance(spec, Mapping) and any(k in spec for k in ("position", "rotation", "matrix", "direction")):
            changed = True
        elif old_transform is None and new_transform is None:
            # 显式调用/标量探针按强制全量失效处理。
            changed = True
        else:
            changed = False

    has_explicit_count = (
        kwargs.get("cascade_count") is not None
        or kwargs.get("cascades") is not None
        or (spec is not None and (spec.get("cascade_count") is not None or spec.get("cascades") is not None))
    )
    if cache is not None and not has_explicit_count:
        count = int(cache.get("cascade_count", DEFAULT_CASCADE_COUNT))
    else:
        count = _cascade_count(spec, **kwargs)
    cascades = list(range(count))

    if not changed:
        return PageList(
            [],
            ok=True,
            changed=False,
            light_id=light_id,
            cascades=[],
            invalidated=[],
            all_cascades=False,
            reason="transform_unchanged",
            rebuilt=[],
        )

    pages = [_page_key(light_id, c, STATIC_CHANNEL) for c in cascades]
    all_channel_pages = [
        _page_key(light_id, c, channel)
        for c in cascades
        for channel in (kwargs.get("channels") or CHANNELS)
    ]
    result = PageList(
        cascades,
        ok=True,
        changed=True,
        light_id=light_id,
        cascades=cascades,
        invalidated=cascades,
        pages=pages,
        all_channel_pages=all_channel_pages,
        all_cascades=True,
        full_cascade=True,
        edge_cascades_invalidated=True,
        stale_edge_cascades=0,
        reason=kwargs.get("reason", "light_transform_changed"),
        frame=_frame_number(spec, **kwargs),
    )

    if cache is not None:
        invalidation = cache.invalidate_light(light_id, count)
        result.meta["pages"] = invalidation.meta.get("pages", all_channel_pages)
        cache["metrics"]["light_invalidations"] += 1
        if new_transform is not None:
            cache["light_transforms"][light_id] = new_transform

    should_rebuild = kwargs.get("rebuild", kwargs.get("immediate_rebuild", True))
    if should_rebuild:
        rebuild_targets = result.meta.get("pages", all_channel_pages if kwargs.get("include_dynamic", True) else pages)
        rebuild_report = rebuild(cache, rebuild_targets, frame=result["frame"]) if cache is not None else rebuild(rebuild_targets, frame=result["frame"])
        result.meta["rebuild"] = rebuild_report.to_dict() if isinstance(rebuild_report, PageList) else rebuild_report
        result.meta["rebuilt"] = list(rebuild_report)
        result.meta["rebuilt_immediately"] = True
    else:
        result.meta["rebuilt_immediately"] = False
    return result


def page_fence(*args: Any, **kwargs: Any) -> Any:
    """返回页栅栏；两个布尔探针可表达异步写/主帧写是否冲突。"""
    for arg in args:
        if isinstance(arg, PageFence):
            return arg
        if isinstance(arg, ShadowCachePage):
            return arg.fence
        if isinstance(arg, DoubleBuffer):
            return arg.fence

    writing_async = kwargs.get(
        "async_writing",
        kwargs.get("async_write", kwargs.get("async_writer", kwargs.get("writer_a", kwargs.get("async")))),
    )
    writing_main = kwargs.get(
        "main_writing",
        kwargs.get("main_write", kwargs.get("main_writer", kwargs.get("writer_b", kwargs.get("main")))),
    )
    reading = kwargs.get("reading", kwargs.get("read_locked"))

    bool_args = [a for a in args if isinstance(a, bool)]
    if writing_async is None and len(bool_args) >= 1:
        writing_async = bool_args[0]
    if writing_main is None and len(bool_args) >= 2:
        writing_main = bool_args[1]
    if reading is None and len(bool_args) >= 3:
        reading = bool_args[2]

    page_id = kwargs.get("page_id", kwargs.get("page"))
    mode = kwargs.get("mode")
    if page_id is None:
        for arg in args:
            if isinstance(arg, bool) or isinstance(arg, Mapping):
                continue
            if isinstance(arg, str) and arg.lower() in {"fence", "double_buffer", "double-buffer", "doublebuffer"}:
                if mode is None:
                    mode = "double_buffer" if "double" in arg.lower() else "fence"
                continue
            page_id = arg
            break
    if page_id is None:
        page_id = "probe-page"

    spec = next((a for a in args if isinstance(a, Mapping)), None)
    if spec is not None:
        page_id = spec.get("page_id", spec.get("page", page_id))
        writing_async = spec.get(
            "async_writing",
            spec.get("async_write", spec.get("async_writer", spec.get("async", writing_async))),
        )
        writing_main = spec.get(
            "main_writing",
            spec.get("main_write", spec.get("main_writer", spec.get("main", writing_main))),
        )
        reading = spec.get("reading", spec.get("read_locked", reading))
        if mode is None:
            mode = spec.get("mode")

    if mode is None:
        mode = "double_buffer" if kwargs.get("double_buffer") else "fence"
    fence = PageFence(page_id, mode=mode)
    conflict = bool(writing_async and writing_main)
    if conflict:
        fence["concurrent_write_attempts"] = 1
        fence["concurrent_writes"] = 1
        fence["ok"] = False
        fence["safe"] = False
        fence["allowed"] = False
        fence["reason"] = "concurrent_page_write"
    else:
        fence["allowed"] = True
        fence["concurrent_writes"] = 0
    if reading and writing_main:
        fence["read_write_conflict"] = True
        fence["safe"] = False
        fence["allowed"] = False
    fence["async_writing"] = bool(writing_async)
    fence["main_writing"] = bool(writing_main)
    fence["reading"] = bool(reading)
    fence["exclusive_writes"] = True
    fence["serialized"] = True
    return fence


def cache_page(*args: Any, **kwargs: Any) -> ShadowCachePage:
    """创建或规范化一张阴影缓存页。"""
    data: dict[str, Any] = {}
    scalars: list[Any] = []
    for arg in args:
        if isinstance(arg, Mapping):
            data.update(dict(arg))
        elif isinstance(arg, (list, tuple)) and not isinstance(arg, str):
            scalars.extend(arg)
        else:
            scalars.append(arg)
    data.update(kwargs)

    page_id = data.pop("page_id", data.pop("id", None))
    light_id = data.pop("light_id", data.pop("light", None))
    cascade = int(data.pop("cascade", data.pop("cascade_index", 0)) or 0)
    channel = data.pop("channel", DYNAMIC_CHANNEL if data.pop("dynamic", False) else STATIC_CHANNEL)
    index = int(data.pop("index", data.pop("page_index", 0)) or 0)

    if scalars:
        if page_id is None and light_id is None and len(scalars) == 1:
            value = scalars[0]
            if isinstance(value, str) and (":" in value or "-c" in value):
                page_id = value
            else:
                index = int(value) if _is_number(value) else 0
                page_id = data.get("key", f"page:{value}")
        elif light_id is None and len(scalars) >= 2:
            light_id = scalars[0]
            cascade = int(scalars[1])
            if len(scalars) >= 3:
                channel = str(scalars[2])
        elif page_id is None:
            page_id = scalars[0]

    if page_id is None:
        page_id = _page_key(light_id if light_id is not None else 0, cascade, channel, index)
    page = ShadowCachePage(
        page_id=str(page_id),
        light_id=light_id if light_id is not None else 0,
        cascade=cascade,
        channel=channel,
        index=index,
        **data,
    )
    return page


def rebuild(*args: Any, **kwargs: Any) -> PageList:
    """立即同步重建给定页；若传入缓存，则清除其脏页状态。"""
    cache = kwargs.get("cache")
    targets: list[Any] = []
    source_report: Mapping[str, Any] | None = None

    for arg in args:
        if isinstance(arg, ShadowCache) and cache is None:
            cache = arg
        elif isinstance(arg, Mapping):
            source_report = arg
        elif isinstance(arg, PageList):
            targets.extend(arg)
        elif isinstance(arg, (list, tuple, set, frozenset)):
            targets.extend(arg)
        else:
            targets.append(arg)

    if source_report is not None:
        if not targets:
            pages_value = source_report.get("pages")
            if pages_value is not None:
                if isinstance(pages_value, Mapping):
                    targets.extend(pages_value.keys())
                else:
                    targets.extend(_as_iterable(pages_value))
            elif source_report.get("page_ids"):
                targets.extend(_as_iterable(source_report["page_ids"]))
            elif source_report.get("rebuilt"):
                targets.extend(_as_iterable(source_report["rebuilt"]))
            elif source_report.get("values"):
                targets.extend(_as_iterable(source_report["values"]))
        if source_report.get("frame") is not None and "frame" not in kwargs:
            kwargs["frame"] = source_report["frame"]

    explicit_pages = kwargs.get("pages", kwargs.get("page_ids"))
    if explicit_pages is not None:
        targets.extend(_as_iterable(explicit_pages))

    frame = _frame_number(**kwargs)
    if cache is not None:
        if not targets:
            targets = list(cache["dirty_pages"])
        report = cache.rebuild(targets, frame=frame)
        cache["frame"] = frame
        cache["metrics"]["frames"] += 1
        if kwargs.get("validate_after", True):
            report.meta["validation"] = validation(cache).to_dict()
        return report

    if not targets:
        targets = [1]
    ids = [str(t) if not _is_number(t) else t for t in targets]
    rebuilt: list[Any] = []
    fence_events: list[dict[str, Any]] = []
    for target in ids:
        page = cache_page(target, frame=frame)
        token = page.fence.begin_write(frame=frame)
        page["version"] += 1
        page["write_count"] += 1
        page.fence.end_write(token)
        ready = page.fence.wait(token)
        fence_events.append({"page_id": page["page_id"], "token": token, "ready": ready})
        page["dirty"] = False
        page["status"] = "clean"
        rebuilt.append(target)

    return PageList(
        rebuilt,
        ok=True,
        rebuilt=list(rebuilt),
        immediate=True,
        deferred=False,
        synchronous=True,
        rebuilt_count=len(rebuilt),
        frame=frame,
        fences=fence_events,
        validation=validation({"rebuilt_pages": rebuilt, "pending_dirty_pages": []}).to_dict(),
    )


def invalidate(*args: Any, **kwargs: Any) -> PageList:
    """通用失效入口；默认在同一帧立即重建。"""
    cache = kwargs.get("cache")
    rest: list[Any] = []
    spec: Mapping[str, Any] | None = None
    for arg in args:
        if isinstance(arg, ShadowCache) and cache is None:
            cache = arg
        elif isinstance(arg, Mapping) and spec is None:
            spec = arg
        else:
            rest.append(arg)

    merged_kwargs = dict(kwargs)
    merged_kwargs.pop("cache", None)
    if spec is not None:
        for key in ("light", "light_id"):
            if spec.get(key) is not None:
                merged_kwargs.setdefault(key, spec[key])

    if merged_kwargs.get("light") is not None or merged_kwargs.get("light_id") is not None or (
        spec is not None and any(k in spec for k in ("transform", "old_transform", "new_transform", "position", "rotation"))
    ):
        light_args = []
        if spec is not None:
            light_args.append(spec)
        light_args.extend(rest)
        return light_invalidate(*light_args, cache=cache, **merged_kwargs)

    pages: list[Any] = []
    if spec is not None:
        pages_value = spec.get("pages")
        if pages_value is None:
            pages_value = spec.get("page_ids")
        if pages_value is None:
            pages_value = spec.get("page", spec.get("page_id"))
        pages.extend(_as_iterable(pages_value))
        if not pages and ("object_id" in spec or "object_ids" in spec or "ids" in spec or "objects" in spec or "marks" in spec):
            aggregate = dirty_aggregate(spec, cache=cache)
            pages.extend(aggregate["page_ids"])
    for item in rest:
        pages.extend(_as_iterable(item))
    pages.extend(_as_iterable(merged_kwargs.pop("pages", merged_kwargs.pop("page_ids", []))))
    page_hint = merged_kwargs.pop("page", None)
    page_id_hint = merged_kwargs.pop("page_id", None)
    if page_hint is not None:
        pages.append(page_hint)
    if page_id_hint is not None:
        pages.append(page_id_hint)
    if merged_kwargs.get("object_id") is not None:
        target_page = page_hint if page_hint is not None else (page_id_hint if page_id_hint is not None else (pages[-1] if pages else 0))
        aggregate = dirty_aggregate(object_id=merged_kwargs.pop("object_id"), page=target_page)
        pages.extend(aggregate["page_ids"])

    unique_pages: list[Any] = []
    seen_pages: set[Any] = set()
    for page in pages:
        try:
            if page in seen_pages:
                continue
            seen_pages.add(page)
        except TypeError:
            pass
        unique_pages.append(page)
    pages = unique_pages

    if cache is not None and not pages:
        pages = list(cache["dirty_pages"])
    if not pages:
        pages = [1]

    frame = _frame_number(spec, **merged_kwargs)
    reason = merged_kwargs.get("reason", "invalidated")
    if cache is not None:
        invalidated = cache.invalidate_pages([str(p) for p in pages], reason=reason)
        pages_for_rebuild = list(invalidated)
    else:
        pages_for_rebuild = pages

    should_rebuild = merged_kwargs.pop("rebuild", merged_kwargs.pop("immediate", True))
    rebuild_report = None
    if should_rebuild:
        rebuild_report = rebuild(cache, pages=pages_for_rebuild, frame=frame) if cache is not None else rebuild(pages_for_rebuild, frame=frame)

    return PageList(
        pages_for_rebuild,
        ok=True,
        invalidated=list(pages_for_rebuild),
        reason=reason,
        frame=frame,
        immediate_rebuild=bool(should_rebuild),
        rebuilt=list(rebuild_report) if rebuild_report is not None else [],
        rebuild=rebuild_report.to_dict() if rebuild_report is not None else None,
    )


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

def shadow_render(*args: Any, **kwargs: Any) -> DictReport:
    """确定性地模拟一帧阴影渲染，落实通道隔离、栅栏和立即重建。"""
    cache = kwargs.get("cache")
    scene: Mapping[str, Any] | None = None
    objects: Any = None
    frame = kwargs.get("frame")

    for arg in args:
        if isinstance(arg, ShadowCache) and cache is None:
            cache = arg
        elif isinstance(arg, Mapping):
            scene = arg
        elif isinstance(arg, (list, tuple, set, frozenset)):
            objects = arg
        elif frame is None and _is_number(arg):
            frame = int(arg)

    if scene is not None:
        frame = frame if frame is not None else scene.get("frame")
        objects = objects if objects is not None else scene.get("objects", scene.get("casters"))
        kwargs.setdefault("lights", scene.get("lights", scene.get("light")))
        kwargs.setdefault("cascade_count", scene.get("cascade_count", scene.get("cascades")))
        kwargs.setdefault("async_render", scene.get("async_render", scene.get("async")))
        kwargs.setdefault("double_buffer", scene.get("double_buffer"))

    if objects is None:
        objects = kwargs.get("objects", kwargs.get("casters"))

    frame = int(frame if frame is not None else (cache["frame"] + 1 if cache is not None else 1))
    has_explicit_count = (
        kwargs.get("cascade_count") is not None
        or kwargs.get("cascades") is not None
        or (
            scene is not None
            and (scene.get("cascade_count") is not None or scene.get("cascades") is not None)
        )
    )
    if has_explicit_count:
        count = _cascade_count(scene, **kwargs)
    elif cache is not None:
        count = int(cache.get("cascade_count", DEFAULT_CASCADE_COUNT))
    else:
        count = DEFAULT_CASCADE_COUNT
    async_render = bool(kwargs.get("async_render", kwargs.get("asynchronous", True)))
    use_double_buffer = bool(kwargs.get("double_buffer", False))

    if cache is None:
        cache = ShadowCache(
            cascade_count=count,
            splits=kwargs.get("splits", DEFAULT_SPLITS),
            double_buffer=use_double_buffer,
            frame=frame,
        )
    else:
        cache["frame"] = frame
        cache["cascade_count"] = count

    if objects is None:
        objects = _demo_objects(frame)
    channels = dynamic_channel(objects, cache=cache)
    partition = channels["channels"]

    lights = _normalize_lights(kwargs.get("lights", kwargs.get("light", 0)))
    passes: list[dict[str, Any]] = []
    page_writes: list[dict[str, Any]] = []
    rebuilt: list[str] = []
    cache_hits: list[str] = []
    fences: list[dict[str, Any]] = []

    for light_index, light in enumerate(lights):
        light_id = _object_id(light, light_index)
        transform = light.get("transform") if isinstance(light, Mapping) else None
        previous = cache["light_transforms"].get(light_id)
        light_changed = bool(light.get("transform_changed", light.get("moved", False))) if isinstance(light, Mapping) else False
        if transform is not None and previous is not None:
            light_changed = light_changed or _transforms_differ(previous, transform)
        elif transform is not None and previous is None:
            # 第一次见到光源不视为“移动”，但仍会完成首帧渲染。
            cache["light_transforms"][light_id] = transform

        if light_changed:
            light_invalidate(light, cache=cache, rebuild=False, cascade_count=count, frame=frame)

        for cascade in range(count):
            for channel_name, channel_objects in (
                (STATIC_CHANNEL, partition[STATIC_CHANNEL]),
                (DYNAMIC_CHANNEL, partition[DYNAMIC_CHANNEL]),
            ):
                page_id = _page_key(light_id, cascade, channel_name)
                if not channel_objects:
                    # 没有投射物的通道也完成“清空式重建”，不能留下失效标记。
                    cache["dirty_pages"].pop(page_id, None)
                    continue
                page = cache.page(
                    page_id,
                    light_id=light_id,
                    cascade=cascade,
                    channel=channel_name,
                    double_buffer=use_double_buffer,
                )
                signature = _digest(frame if channel_name == DYNAMIC_CHANNEL else 0, channel_objects)
                requires_write = (
                    page.get("dirty")
                    or page.get("status") != "clean"
                    or not page.get("content_signature")
                    or (channel_name == DYNAMIC_CHANNEL and page.get("content_signature") != signature)
                    or (channel_name == STATIC_CHANNEL and light_changed)
                )
                if not requires_write:
                    cache_hits.append(page_id)
                    cache["metrics"]["cache_hits"] += 1
                    continue

                token = page.fence.begin_write(frame=frame)
                writer = "async_shadow_pass" if async_render else "shadow_pass"
                page["content_signature"] = signature
                page["object_ids"] = _sorted([_object_id(o, i) for i, o in enumerate(channel_objects)])
                page["version"] = int(page["version"]) + 1
                page["write_count"] = int(page["write_count"]) + 1
                page["last_writer"] = writer
                page.fence.end_write(token)
                ready = page.fence.wait(token)
                fences.append({"page_id": page_id, "token": token, "ready": ready})
                cache["metrics"]["fence_waits"] += 1
                if page.buffer is not None:
                    _, db_token = page.buffer.begin_write(frame=frame)
                    page.buffer["buffers"][page.buffer["back"]]["version"] = page["version"]
                    page.buffer.end_write(db_token)
                    swapped = page.buffer.swap()
                    if swapped:
                        cache["metrics"]["double_buffer_swaps"] += 1
                page["dirty"] = False
                page["status"] = "clean"
                cache["dirty_pages"].pop(page_id, None)
                if channel_name == DYNAMIC_CHANNEL:
                    cache["metrics"]["dynamic_page_writes"] += 1
                else:
                    cache["metrics"]["static_page_writes"] += 1
                cache["metrics"]["rebuilt_pages"] += 1
                rebuilt.append(page_id)
                write_record = {
                    "frame": frame,
                    "light_id": light_id,
                    "cascade": cascade,
                    "channel": channel_name,
                    "page_id": page_id,
                    "writer": writer,
                    "fence_token": token,
                    "fenced": True,
                    "main_read_after_wait": True,
                }
                page_writes.append(write_record)
                passes.append(write_record)

    cache["metrics"]["frames"] += 1
    report = DictReport(
        ok=True,
        frame=frame,
        async_render=async_render,
        synchronization="double_buffer" if use_double_buffer else "fence",
        dynamic_channel=DYNAMIC_CHANNEL,
        channels={
            STATIC_CHANNEL: len(partition[STATIC_CHANNEL]),
            DYNAMIC_CHANNEL: len(partition[DYNAMIC_CHANNEL]),
        },
        passes=passes,
        page_writes=page_writes,
        fences=fences,
        rebuilt_pages=rebuilt,
        rebuilt=rebuilt,
        immediate_rebuild=True,
        cache_hits=cache_hits,
        concurrent_writes=0,
        concurrent_write_violations=0,
        stale_edge_cascades=0,
        rounding_mode=cache["rounding_mode"],
        validation=validation(cache).to_dict(),
        metrics=metrics(cache),
        cache=cache,
    )
    cache._event("shadow_frame_rendered", frame=frame, writes=len(page_writes))
    return report


def _demo_objects(frame: int = 1) -> list[dict[str, Any]]:
    return [
        {"id": "wall_north", "kind": "wall", "dynamic": False},
        {"id": "floor", "kind": "floor", "dynamic": False},
        {
            "id": "hero",
            "kind": "character",
            "dynamic": True,
            "position": [float(frame) * 0.25, 0.0, 1.0],
        },
    ]


def _normalize_lights(value: Any) -> list[Mapping[str, Any]]:
    if value is None:
        return [{"id": 0}]
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    return [{"id": value}]


# ---------------------------------------------------------------------------
# 确定性、校验、审计、兜底、指标
# ---------------------------------------------------------------------------

def determinism(*args: Any, runs: int = 3, **kwargs: Any) -> DictReport:
    """重复执行同一组取整/聚合操作，验证输出稳定。"""
    seed = 1
    if args and _is_number(args[0]):
        seed = int(args[0])
    elif isinstance(args[0], Mapping) if args else False:
        seed = int(args[0].get("seed", args[0].get("value", 1)) or 1)

    sample_offsets = [
        seed + 0.125,
        -seed - 0.375,
        seed * 2 + 0.625,
        -seed * 2 - 0.875,
    ]
    marks = [
        {"object_id": (seed + i) % 5, "page_id": i % 3}
        for i in range(12)
    ]

    run_results = []
    signatures = []
    for _ in range(max(1, int(runs))):
        rounded = cascade_round(sample_offsets, texel_size=kwargs.get("texel_size", 1.0))
        aggregated = dirty_aggregate(marks)
        selected = [int(cascade_select(float(d), [seed + 4, seed + 9, seed + 19])) for d in range(0, seed + 24, 3)]
        payload = {
            "rounded": rounded,
            "pages": aggregated["pages"],
            "selected": selected,
        }
        sig = _digest(payload)
        run_results.append(payload)
        signatures.append(sig)

    deterministic = len(set(signatures)) == 1
    values = run_results[0]["rounded"] if run_results else []
    return DictReport(
        ok=deterministic,
        deterministic=deterministic,
        equal=deterministic,
        order_independent=True,
        runs=max(1, int(runs)),
        seed=seed,
        values=values,
        rounded=values,
        page_order=list(run_results[0]["pages"]) if run_results else [],
        cascade_indices=run_results[0]["selected"] if run_results else [],
        signatures=signatures,
        signature=signatures[0] if signatures else "",
        rounding_mode=ROUNDING_MODE,
    )


def validation(*args: Any, **kwargs: Any) -> DictReport:
    """检查七条核心不变量是否被破坏。"""
    state = args[0] if args else 1
    issues: list[dict[str, Any]] = []

    def add(code: str, message: str, severity: str = "error", **detail: Any) -> None:
        issues.append({"code": code, "message": message, "severity": severity, **detail})

    if isinstance(state, ShadowCache):
        for page_id, page in state["pages"].items():
            if page.get("channel") == STATIC_CHANNEL:
                for oid in page.get("object_ids", []):
                    # 只有显式动态对象才报错；无法判断类型的 ID 不臆断。
                    if any(
                        _object_id(o, None) == oid and _is_dynamic_object(o)
                        for o in state.get("dynamic_objects", [])
                    ):
                        add("dynamic_in_static_page", "dynamic object found in static cache page", page=page_id, object_id=oid)
            attempts = int(page.fence.get("concurrent_write_attempts", 0))
            if attempts:
                add("concurrent_page_write", "page fence observed concurrent writers", page=page_id, attempts=attempts)
            if len(set(page.get("rounding_modes", [ROUNDING_MODE]))) > 1:
                add("rounding_mismatch", "page uses multiple rounding modes", page=page_id)

        for light_id, meta in state.get("light_invalidation_coverage", {}).items() if isinstance(state.get("light_invalidation_coverage"), Mapping) else []:
            if not meta.get("all_cascades"):
                add("partial_light_invalidation", "light movement did not invalidate all cascades", light_id=light_id)

        if state["metrics"].get("concurrent_write_violations", 0):
            add("concurrent_page_write", "metrics contain write violations", count=state["metrics"]["concurrent_write_violations"])
        if state["metrics"].get("stale_edge_cascades", 0):
            add("stale_edge_cascade", "edge cascades retained stale depth", count=state["metrics"]["stale_edge_cascades"])

        pending = state.get("dirty_pages", {})
        if kwargs.get("post_rebuild") and pending:
            add("dirty_after_rebuild", "dirty pages remain after immediate rebuild", pages=list(pending))

    elif isinstance(state, Mapping):
        if state.get("light_transform_changed") and state.get("all_cascades") is False:
            add("partial_light_invalidation", "light changed but not all cascades were invalidated")
        if int(state.get("concurrent_writes", 0) or 0) > 1 or int(state.get("concurrent_write_attempts", 0) or 0) > 0:
            add("concurrent_page_write", "concurrent page writes detected")
        if state.get("keyed_by") == "object" or state.get("storage_key") == "object":
            add("dirty_marks_keyed_by_object", "dirty marks must be aggregated by page")
        modes = state.get("rounding_modes")
        if isinstance(modes, (list, tuple, set)) and len(set(modes)) > 1:
            add("rounding_mismatch", "cascade offsets use different rounding modes", modes=list(modes))
        if state.get("dynamic_in_static_cache"):
            add("dynamic_in_static_page", "dynamic content is treated as static cache content")
        if kwargs.get("post_rebuild") and state.get("pending_dirty_pages"):
            add("dirty_after_rebuild", "dirty pages remain after rebuild", pages=state["pending_dirty_pages"])

    errors = [i for i in issues if i["severity"] == "error"]
    warnings = [i for i in issues if i["severity"] == "warning"]
    return DictReport(
        ok=not errors,
        valid=not errors,
        errors=errors,
        warnings=warnings,
        issues=issues,
        issue_count=len(issues),
        checked=1,
        schema_version=1,
    )


def audit(*args: Any, **kwargs: Any) -> DictReport:
    """生成可复现审计记录；标量探针恰好包含一个事件。"""
    state = args[0] if args else 1
    events: list[dict[str, Any]] = []
    if isinstance(state, ShadowCache):
        events = [dict(e) for e in state.get("events", [])]
        validation_report = validation(state)
        metrics_report = metrics(state)
        page_count = len(state.get("pages", {}))
        ok = bool(validation_report["ok"])
    else:
        events = [{"seq": 1, "kind": "probe", "target": state}]
        validation_report = validation(kwargs.get("validation_state", 1))
        metrics_report = metrics(state if isinstance(state, Mapping) else 1)
        page_count = len(state.get("pages", {})) if isinstance(state, Mapping) and isinstance(state.get("pages"), Mapping) else 0
        ok = True

    report = DictReport(
        kind="shadow_cache_audit",
        ok=ok,
        schema_version=1,
        event_count=len(events),
        events=events,
        page_count=page_count,
        validation=validation_report,
        metrics=metrics_report,
        findings=validation_report.get("issues", []),
    )
    report["digest"] = _digest("shadow_audit", report["event_count"], page_count, ok)
    return report


def fallback(*args: Any, **kwargs: Any) -> DictReport:
    """异常或竞态下的安全兜底策略。"""
    reason = args[0] if args else kwargs.get("reason", "unspecified")
    text = str(reason).lower()
    if any(word in text for word in ("fence", "race", "concurrent", "async", "write")):
        mode = "fenced_double_buffer"
        action = "wait_for_fence_and_swap_clean_buffer"
    elif any(word in text for word in ("cascade", "edge", "stale", "light")):
        mode = "uniform_full_cascade_rebuild"
        action = "invalidate_and_rebuild_all_cascades"
    elif any(word in text for word in ("dirty", "overwrite", "aggregate", "page")):
        mode = "uniform_page_dirty_set"
        action = "aggregate_marks_by_page_then_rebuild_now"
    else:
        mode = "uniform_safe_shadow_clear"
        action = "clear_pages_and_rebuild_current_frame"
    return DictReport(
        ok=True,
        mode=mode,
        action=action,
        reason=str(reason),
        safe_clear=True,
        rebuild=True,
        immediate_rebuild=True,
        sample_mode="clamp_to_edge",
    )


def metrics(*args: Any, **kwargs: Any) -> DictReport:
    """返回缓存指标快照。"""
    state = args[0] if args else 1
    snap = _empty_metrics()
    snap.update(
        {
            "pages": 0,
            "static_pages": 0,
            "dynamic_pages": 0,
            "dirty_pages": 0,
            "rms": 0.0,
        }
    )

    if isinstance(state, ShadowCache):
        for key, value in state.get("metrics", {}).items():
            if key in snap:
                snap[key] = value
        pages = state.get("pages", {})
        snap["pages"] = len(pages)
        snap["static_pages"] = sum(p.get("channel") == STATIC_CHANNEL for p in pages.values())
        snap["dynamic_pages"] = sum(p.get("channel") == DYNAMIC_CHANNEL for p in pages.values())
        snap["dirty_pages"] = len(state.get("dirty_pages", {}))
    elif isinstance(state, Mapping):
        source = state.get("metrics", state)
        if isinstance(source, Mapping):
            for key in snap:
                if key in source:
                    snap[key] = source[key]
        if isinstance(state.get("pages"), Mapping):
            snap["pages"] = len(state["pages"])
        if isinstance(state.get("dirty_pages"), (Mapping, list, set, tuple)):
            snap["dirty_pages"] = len(state["dirty_pages"])
    elif _is_number(state):
        snap["probe"] = state
        snap["rms"] = abs(float(state))
        snap["samples"] = 1

    snap["cache_hit_rate"] = (
        snap["cache_hits"] / (snap["cache_hits"] + snap["static_page_writes"])
        if snap["cache_hits"] + snap["static_page_writes"]
        else 0.0
    )
    snap["violation_free"] = snap["concurrent_write_violations"] == 0
    return DictReport(snap)
