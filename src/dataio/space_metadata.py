"""Decode the organizer's 26 doubles stored in the first image row.

Reference: 天基图像数据格式说明.docx, Python appendix. The FITS image
remains signed as declared by BITPIX/BZERO; telemetry is a separate byte payload.
"""
from __future__ import annotations

import struct
import numpy as np

FIELDS = ('p_az', 'p_el', 'ra_deg', 'dec_deg',
          'j2000_x', 'j2000_y', 'j2000_z', 'j2000_xv', 'j2000_yv', 'j2000_zv',
          'q1', 'q2', 'q3', 'q4', 'roll_deg', 'pitch_deg', 'yaw_deg',
          'roll_rate_deg_s', 'pitch_rate_deg_s', 'yaw_rate_deg_s',
          'wgs84_x', 'wgs84_y', 'wgs84_z', 'wgs84_xv', 'wgs84_yv', 'wgs84_zv')


def decode_auxiliary(raw: bytes) -> dict:
    if len(raw) != 208:
        raise ValueError('天基辅助数据必须包含 208 字节')
    swapped = np.frombuffer(raw, dtype=np.uint16).byteswap().tobytes()
    values = np.array(struct.unpack('<26d', swapped))
    radius = np.linalg.norm(values[4:7])
    qnorm = np.linalg.norm(values[10:14])
    if (not np.isfinite(values).all() or not 0 <= values[2] < 360 or
            not -90 <= values[3] <= 90 or not 6e6 < radius < 1e8 or
            not 0.98 < qnorm < 1.02 or not 0 < np.linalg.norm(values[7:10]) < 20000):
        raise ValueError('首行内容未通过天基位置、指向和四元数校验')
    return dict(zip(FIELDS, values.tolist()))
