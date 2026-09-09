"""Resource discovery and conservative worker planning."""

from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ResourceProfile:
    """The resources visible to the process, with zeroes for unavailable GPU data."""

    cpu_logical: int
    ram_total_gib: float
    ram_available_gib: float
    gpu_name: str = "unavailable"
    gpu_total_mib: int = 0
    gpu_free_mib: int = 0
    gpu_used_mib: int = 0
    gpu_util_percent: int = 0


@dataclass(frozen=True)
class ResourcePlan:
    """Bounded settings for a feature extraction run."""

    workers: int
    decode_threads: int
    batch_size: int
    prefetch_batches: int
    pause_below_free_vram_mib: int
    pause_above_gpu_util_percent: int


def _gib(bytes_value: int) -> float:
    return round(bytes_value / 2**30, 2)


def _windows_memory() -> tuple[int, int]:
    try:
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("load", ctypes.c_ulong),
                ("total_phys", ctypes.c_ulonglong),
                ("avail_phys", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("avail_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("avail_virtual", ctypes.c_ulonglong),
                ("avail_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total_phys), int(status.avail_phys)
    except (AttributeError, OSError):
        pass
    return 0, 0


def _gpu_snapshot() -> tuple[str, int, int, int, int]:
    command = ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu",
               "--format=csv,noheader,nounits"]
    try:
        line = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10).stdout.splitlines()[0]
        name, total, free, used, utilisation = (part.strip() for part in line.split(","))
        return name, int(total), int(free), int(used), int(utilisation)
    except (FileNotFoundError, subprocess.SubprocessError, IndexError, ValueError):
        return "unavailable", 0, 0, 0, 0


def discover_resources() -> ResourceProfile:
    """Read host capacity without changing host configuration."""
    total, available = _windows_memory()
    name, gpu_total, gpu_free, gpu_used, gpu_util = _gpu_snapshot()
    return ResourceProfile(os.cpu_count() or 1, _gib(total), _gib(available), name,
                           gpu_total, gpu_free, gpu_used, gpu_util)


def choose_resource_plan(profile: ResourceProfile) -> ResourcePlan:
    """Choose a small, conservative plan from a resource snapshot."""
    enough_ram = not profile.ram_available_gib or profile.ram_available_gib >= 8
    workers = 2 if profile.gpu_free_mib >= 6144 and profile.cpu_logical >= 8 and enough_ram else 1
    batch_size = 4 if profile.gpu_free_mib >= 4096 else 2 if profile.gpu_free_mib >= 2560 else 1
    return ResourcePlan(
        workers=workers,
        decode_threads=max(1, min(4, profile.cpu_logical // 3 or 1)),
        batch_size=batch_size,
        prefetch_batches=2,
        pause_below_free_vram_mib=2048,
        pause_above_gpu_util_percent=95,
    )


def plan_as_dict(profile: ResourceProfile) -> dict[str, object]:
    """Return JSON-ready discovery and planning data."""
    return {"resource_profile": asdict(profile), "resource_plan": asdict(choose_resource_plan(profile))}
