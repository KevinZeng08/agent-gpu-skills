# Versions

NCCL4Py exposes top-level helpers to inspect the installed NCCL stack:
`nccl4py` itself plus the native libraries `libnccl.so` and
`libnccl_ep.so`.

```python
import nccl
nccl.show_versions()      # human-readable block to stdout
v = nccl.get_version()    # programmatic snapshot
```

## show_versions

### nccl.show_versions() → None

Print a summary of the installed NCCL stack to stdout.

For each component, reports the release version, the CUDA toolkit it was
built with, and (for native libraries) the path of the loaded `.so`.

## get_version

### nccl.get_version() → [VersionInfo](#nccl.VersionInfo)

Return a structured snapshot of NCCL stack versions.

* **Returns:**
  [`VersionInfo`](#nccl.VersionInfo) with nccl4py + libnccl + libnccl_ep versions,
  CUDA build variants, and loaded `.so` paths.

## VersionInfo

### *class* nccl.VersionInfo(nccl4py: [Version](https://packaging.pypa.io/en/stable/version.html#packaging.version.Version), nccl: [LibraryInfo](#nccl.LibraryInfo) | None, nccl_ep: [LibraryInfo](#nccl.LibraryInfo) | None)

Bases: `object`

Aggregate version snapshot of the NCCL stack.

#### nccl4py *: [Version](https://packaging.pypa.io/en/stable/version.html#packaging.version.Version)*

nccl4py package version.

#### nccl *: [LibraryInfo](#nccl.LibraryInfo) | None*

Version/CUDA-variant/path of the `libnccl.so` nccl4py is using, or
None when it cannot be loaded.

#### nccl_ep *: [LibraryInfo](#nccl.LibraryInfo) | None*

Version/CUDA-variant/path of the `libnccl_ep.so` nccl4py is using, or
None when it cannot be loaded.

## LibraryInfo

### *class* nccl.LibraryInfo(version: [Version](https://packaging.pypa.io/en/stable/version.html#packaging.version.Version), cuda_variant: [Version](https://packaging.pypa.io/en/stable/version.html#packaging.version.Version) | None, path: Path | None)

Bases: `object`

Version, CUDA build variant, and loaded-path info for a shared library.

#### version *: [Version](https://packaging.pypa.io/en/stable/version.html#packaging.version.Version)*

Library release version (e.g. `2.30.0`).

#### cuda_variant *: [Version](https://packaging.pypa.io/en/stable/version.html#packaging.version.Version) | None*

CUDA toolkit major.minor the library was built with (e.g. `12.9`), or
None if it could not be read from the library.

#### path *: Path | None*

Path of the loaded `libnccl.so`, or None if it cannot be determined.
