"""Prefer target-platform installed packages; allow only the matching legacy vendor ABI."""
from pathlib import Path
import platform
import sys
import sysconfig


def vendored_runtime_matches():
    return (sys.implementation.name == 'cpython'
            and sys.version_info[:2] == (3, 12)
            and sys.platform == 'linux'
            and platform.machine().lower() in ('aarch64', 'arm64')
            and sysconfig.get_config_var('SOABI') == 'cpython-312-aarch64-linux-gnu')


def configure_dependencies(project):
    vendor = str((Path(project)/'vendor').resolve())
    # Never let the old platform bundle shadow a correctly installed package.
    sys.path[:] = [p for p in sys.path if p != vendor]
    enabled = Path(vendor).is_dir() and vendored_runtime_matches()
    if enabled:
        sys.path.append(vendor)
    return {'legacy_vendor_enabled': enabled,
            'installed_packages_preferred': True,
            'legacy_vendor_abi': 'cpython-312-aarch64-linux-gnu'}
