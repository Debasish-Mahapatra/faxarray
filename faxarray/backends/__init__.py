"""Backend implementations for FA/LFI access."""

from .native_fa import NativeFAResource, write_fa

__all__ = ["NativeFAResource", "write_fa"]
