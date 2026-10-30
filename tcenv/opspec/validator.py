"""A closed JSON Schema subset for registered semantic documents.

The plan (§4.1) requires every OpSpec/track/LoweringProfile document to be checked
against a *registered JSON Schema*, with unknown fields refused at every level.  The
repository has no ``jsonschema`` dependency, so this module implements exactly the
keyword subset the registered schemas use and **refuses any keyword it does not
implement** — a schema can therefore never contain a silently unenforced rule.

Supported keywords: ``type`` (string or list), ``enum``, ``const``, ``required``,
``properties``, ``additionalProperties`` (bool or schema), ``items``, ``minItems``,
``maxItems``, ``minLength``, ``maxLength``, ``pattern``, ``minimum``, ``maximum``,
``exclusiveMinimum``, ``exclusiveMaximum``, ``minProperties``, ``maxProperties``,
``anyOf``, ``allOf``, ``$comment``, ``title``, ``description``.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

_SUPPORTED = frozenset(
    {
        "type",
        "enum",
        "const",
        "required",
        "properties",
        "additionalProperties",
        "items",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "pattern",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minProperties",
        "maxProperties",
        "anyOf",
        "allOf",
        "$comment",
        "title",
        "description",
    }
)

_JSON_TYPES = frozenset({"object", "array", "string", "number", "integer", "boolean", "null"})


class SchemaFailure(ValueError):
    """@brief 文档不符合登记 JSON Schema。"""


def _type_name(value: Any) -> str:
    """@brief 返回一个值对应的 JSON 类型名（bool 不是 integer/number）。

    @param value 任意值。
    @return JSON 类型名。
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    return type(value).__name__


def _matches_type(value: Any, name: str) -> bool:
    """@brief 判断值是否属于某个 JSON 类型（``number`` 接受整数）。

    @param value 待检查的值。
    @param name JSON 类型名。
    @return 匹配时为 True。
    """
    kind = _type_name(value)
    if name == "number":
        return kind in ("integer", "number")
    if name == "integer":
        return kind == "integer"
    return kind == name


def _check_keyword(keyword: str, where: str) -> None:
    """@brief 拒绝未实现的 JSON Schema 关键字。

    @param keyword 关键字名。
    @param where 该 schema 所在位置的诊断路径。
    @raises SchemaFailure 关键字未登记。
    """
    if keyword not in _SUPPORTED:
        raise SchemaFailure(f"{where}: unsupported JSON Schema keyword {keyword!r}")


def validate(value: Any, schema: Mapping[str, Any], *, path: str = "$") -> None:
    """@brief 按登记 JSON Schema 校验一个值。

    @param value 待校验的 Python 值（已由严格 JSON loader 解析）。
    @param schema 登记的 schema 片段。
    @param path 诊断路径，顶层为 ``$``。
    @raises SchemaFailure 值不符合 schema，或 schema 使用了未登记关键字。
    """
    if not isinstance(schema, Mapping):
        raise SchemaFailure(f"{path}: schema must be an object")
    for keyword in schema:
        _check_keyword(keyword, f"{path} (schema)")

    if "anyOf" in schema:
        errors = []
        for index, option in enumerate(schema["anyOf"]):
            try:
                validate(value, option, path=path)
                break
            except SchemaFailure as exc:  # noqa: PERF203 - every branch must be tried
                errors.append(f"[{index}] {exc}")
        else:
            raise SchemaFailure(f"{path}: no anyOf branch matched: {'; '.join(errors)}")
    if "allOf" in schema:
        for index, option in enumerate(schema["allOf"]):
            validate(value, option, path=f"{path} (allOf[{index}])")

    expected = schema.get("type")
    if expected is not None:
        options = [expected] if isinstance(expected, str) else list(expected)
        for name in options:
            if name not in _JSON_TYPES:
                raise SchemaFailure(f"{path}: unsupported JSON type {name!r} in schema")
        if not any(_matches_type(value, name) for name in options):
            raise SchemaFailure(f"{path}: expected {'/'.join(options)}, got {_type_name(value)}")

    if "const" in schema and value != schema["const"]:
        raise SchemaFailure(f"{path}: expected the constant {schema['const']!r}, got {value!r}")
    if "enum" in schema:
        if not any(value == option and _type_name(value) == _type_name(option) for option in schema["enum"]):
            raise SchemaFailure(f"{path}: {value!r} is not one of {list(schema['enum'])!r}")

    if isinstance(value, str):
        _validate_string(value, schema, path)
    elif isinstance(value, int) and not isinstance(value, bool):
        _validate_number(value, schema, path)
    elif isinstance(value, float):
        _validate_number(value, schema, path)
    elif isinstance(value, (list, tuple)):
        _validate_array(value, schema, path)
    elif isinstance(value, Mapping):
        _validate_object(value, schema, path)


def _validate_string(value: str, schema: Mapping[str, Any], path: str) -> None:
    """@brief 校验字符串的 ``minLength``/``maxLength``/``pattern``。

    @param value 字符串值。
    @param schema schema 片段。
    @param path 诊断路径。
    @raises SchemaFailure 不满足约束。
    """
    if "minLength" in schema and len(value) < schema["minLength"]:
        raise SchemaFailure(f"{path}: string is shorter than {schema['minLength']}")
    if "maxLength" in schema and len(value) > schema["maxLength"]:
        raise SchemaFailure(f"{path}: string is longer than {schema['maxLength']}")
    if "pattern" in schema and not re.fullmatch(schema["pattern"], value):
        raise SchemaFailure(f"{path}: {value!r} does not match {schema['pattern']!r}")


def _validate_number(value: int | float, schema: Mapping[str, Any], path: str) -> None:
    """@brief 校验数值的上下界约束。

    @param value 数值。
    @param schema schema 片段。
    @param path 诊断路径。
    @raises SchemaFailure 越界。
    """
    if "minimum" in schema and value < schema["minimum"]:
        raise SchemaFailure(f"{path}: {value!r} is below the minimum {schema['minimum']!r}")
    if "maximum" in schema and value > schema["maximum"]:
        raise SchemaFailure(f"{path}: {value!r} is above the maximum {schema['maximum']!r}")
    if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
        raise SchemaFailure(f"{path}: {value!r} must be greater than {schema['exclusiveMinimum']!r}")
    if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
        raise SchemaFailure(f"{path}: {value!r} must be less than {schema['exclusiveMaximum']!r}")


def _validate_array(value: Any, schema: Mapping[str, Any], path: str) -> None:
    """@brief 校验数组长度与元素。

    @param value 数组值。
    @param schema schema 片段。
    @param path 诊断路径。
    @raises SchemaFailure 不满足约束。
    """
    if "minItems" in schema and len(value) < schema["minItems"]:
        raise SchemaFailure(f"{path}: fewer than {schema['minItems']} items")
    if "maxItems" in schema and len(value) > schema["maxItems"]:
        raise SchemaFailure(f"{path}: more than {schema['maxItems']} items")
    items = schema.get("items")
    if items is not None:
        for index, item in enumerate(value):
            validate(item, items, path=f"{path}[{index}]")


def _validate_object(value: Mapping[str, Any], schema: Mapping[str, Any], path: str) -> None:
    """@brief 校验对象必填键、未知键与各字段。

    @param value 对象值。
    @param schema schema 片段。
    @param path 诊断路径。
    @raises SchemaFailure 缺必填键、出现未知键或字段非法。
    """
    if "minProperties" in schema and len(value) < schema["minProperties"]:
        raise SchemaFailure(f"{path}: fewer than {schema['minProperties']} properties")
    if "maxProperties" in schema and len(value) > schema["maxProperties"]:
        raise SchemaFailure(f"{path}: more than {schema['maxProperties']} properties")
    for key in schema.get("required", ()):
        if key not in value:
            raise SchemaFailure(f"{path}: missing required field {key!r}")
    properties = schema.get("properties", {})
    additional = schema.get("additionalProperties", True)
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise SchemaFailure(f"{path}: field names must be non-empty strings")
        if key in properties:
            validate(item, properties[key], path=f"{path}.{key}")
            continue
        if additional is False:
            raise SchemaFailure(f"{path}: unknown field {key!r}")
        if isinstance(additional, Mapping):
            validate(item, additional, path=f"{path}.{key}")


__all__ = ["SchemaFailure", "validate"]
