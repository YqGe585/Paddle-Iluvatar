"""
Paddle-Iluvatar: Iluvatar GPU adapter for PaddlePaddle

This package provides device management, memory operations, and stream
management for Iluvatar GPUs to work with the PaddlePaddle framework.
"""

__version__ = '0.1.0'

from . import device
from . import memory
from . import stream

__all__ = [
    'device',
    'memory', 
    'stream',
]
