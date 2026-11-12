"""fp32 scratch as a linear ``R32Float`` texture view of the same guarded buffer.

This Metal toolchain reports ``MTLBindingAccessReadWrite`` for every non-const
``device`` buffer, including ``u[tid] = 1.0f``.  ``texture2d<float, access::write>``
reflects ``write`` and ``access::read`` reflects ``read``.  The first
``uninitialized_poison`` producer therefore has to be a write-only texture if the
lifecycle check is allowed to trust reflection.

The view is a tight 2D packing of the logical elements plus one extra row.  That
row starts at the first byte of the tail guard, so a store one element past the
declared scratch still corrupts the ``0xCD`` guard.  ``bytesPerRow`` is 16-byte
aligned (the linear-texture rule on this GPU) and the extra row is never wider
than the guard.
"""

from __future__ import annotations

from typing import Any

from ...contract_errors import GeometryError

_MAX_DIM = 16384
_ELEM = 4
_ROW_ALIGN = 16


def scratch_texture_layout(count: int, *, guard_bytes: int) -> tuple[int, int, int]:
    """@brief 选择覆盖 ``count`` 个 fp32 的纹理尺寸，并多留一整行给尾保护带。

    @param count 逻辑元素个数。
    @param guard_bytes 尾保护带字节数；额外一行必须落在里面。
    @return ``(width, height, bytes_per_row)``。``height`` 含额外的保护行，
        逻辑元素是前 ``height - 1`` 行。
    @raises GeometryError 没有同时满足对齐、维度上限和保护带的宽度。
    """
    if count < 1:
        raise GeometryError("a scratch texture needs at least one fp32 element")
    if guard_bytes < _ROW_ALIGN:
        raise GeometryError(f"scratch texture guard {guard_bytes} is shorter than a 16-byte row")
    max_width = min(_MAX_DIM, guard_bytes // _ELEM)
    max_width -= max_width % (_ROW_ALIGN // _ELEM)
    if max_width < _ROW_ALIGN // _ELEM:
        raise GeometryError("the tail guard cannot hold one aligned scratch-texture row")
    # One row is reserved for the guard, so the data rows stay inside the texture limit.
    data_height_limit = _MAX_DIM - 1
    min_width = (count + data_height_limit - 1) // data_height_limit
    align = _ROW_ALIGN // _ELEM
    width = max(align, ((min_width + align - 1) // align) * align)
    while width <= max_width:
        if count % width == 0:
            data_height = count // width
            row_bytes = width * _ELEM
            if data_height <= data_height_limit and row_bytes <= guard_bytes and row_bytes % _ROW_ALIGN == 0:
                return width, data_height + 1, row_bytes
        width += align
    raise GeometryError(
        f"scratch of {count} fp32 elements has no texture layout whose extra row stays inside "
        f"the {guard_bytes}-byte guard"
    )


def make_scratch_texture(buffer: Any, *, offset: int, count: int, guard_bytes: int) -> Any:
    """@brief 在已分配的 scratch 缓冲上建 ``R32Float`` 视图。

    @param buffer ``MTLBuffer``（逻辑区从 ``offset`` 开始）。
    @param offset 逻辑区起点；额外一行从逻辑区末尾进入尾保护带。
    @param count 逻辑 fp32 个数。
    @param guard_bytes 尾保护带字节数。
    @return 纹理。调用方持有它，直到对应的缓冲释放。
    @raises GeometryError 布局不存在。
    """
    import Metal

    width, height, row_bytes = scratch_texture_layout(count, guard_bytes=guard_bytes)
    desc = Metal.MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
        Metal.MTLPixelFormatR32Float, width, height, False
    )
    desc.setUsage_(Metal.MTLTextureUsageShaderRead | Metal.MTLTextureUsageShaderWrite)
    desc.setStorageMode_(Metal.MTLStorageModeShared)
    texture = buffer.newTextureWithDescriptor_offset_bytesPerRow_(desc, int(offset), int(row_bytes))
    if texture is None:
        raise GeometryError(
            f"Metal refused the scratch texture view ({width}x{height}, offset {offset}, stride {row_bytes})"
        )
    return texture
