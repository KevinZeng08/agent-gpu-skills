# Library Constants

The NVSHMEM library provides a set of compile-time constants that may be used to specify options to API routines, provide implementation-specific parameters, or return information about the implementation.

`NVSHMEM_THREAD_SINGLE`

The NVSHMEM thread support level which specifies that the program must not be multithreaded. See Section Thread Support for more detail about its use.

`NVSHMEM_THREAD_FUNNELED`

The NVSHMEM thread support level which specifies that the program may be multithreaded but must ensure that only the main thread invokes the NVSHMEM interfaces. See Section Thread Support for more detail about its use.

`NVSHMEM_THREAD_SERIALIZED`

The NVSHMEM thread support level which specifies that the program may be multithreaded but must ensure that the NVSHMEM interfaces are not invoked concurrently by multiple threads. See Section Thread Support for more detail about its use.

`NVSHMEM_THREAD_MULTIPLE`

The NVSHMEM thread support level which specifies that the program may be multithreaded and any thread may invoke the NVSHMEM interfaces. See Section Thread Support for more detail about its use.

`NVSHMEM_TEAM_INVALID`

A value corresponding to an invalid team. This value can be used to initialize or update team handles to indicate that they do not reference a valid team. When managed in this way, applications can use an equality comparison to test whether a given team handle references a valid team. See Section Team Management for more detail about its use.

`NVSHMEM_SIGNAL_SET`

An integer constant expression corresponding to the signal update set operation. See Section NVSHMEM_PUT_SIGNAL and Section NVSHMEM_PUT_SIGNAL_NBI for more detail about its use.

`NVSHMEM_SIGNAL_ADD`

An integer constant expression corresponding to the signal update add operation. See Section NVSHMEM_PUT_SIGNAL and Section NVSHMEM_PUT_SIGNAL_NBI for more detail about its use.

`NVSHMEM_MAJOR_VERSION`

Integer representing the major version of OpenSHMEM Specification in use.

`NVSHMEM_MINOR_VERSION`

Integer representing the minor version of OpenSHMEM Specification in use.

`NVSHMEM_MAX_NAME_LEN`

Integer representing the maximum length of `NVSHMEM_VENDOR_STRING`.

`NVSHMEM_CMP_EQ`

An integer constant expression corresponding to the “equal to” comparison operation. See Section Point-To-Point Synchronization for more detail about its use.

`NVSHMEM_CMP_NE`

An integer constant expression corresponding to the “not equal to” comparison operation. See Section Point-To-Point Synchronization for more detail about its use.

`NVSHMEM_CMP_LT`

An integer constant expression corresponding to the “less than” comparison operation. See Section Point-To-Point Synchronization for more detail about its use.

`NVSHMEM_CMP_LE`

An integer constant expression corresponding to the “less than or equal to” comparison operation. See Section Point-To-Point Synchronization for more detail about its use.

`NVSHMEM_CMP_GT`

An integer constant expression corresponding to the “greater than” comparison operation. See Section Point-To-Point Synchronization for more detail about its use.

`NVSHMEM_CMP_GE`

An integer constant expression corresponding to the “greater than or equal to” comparison operation. See Section Point-To-Point Synchronization for more detail about its use.

`NVSHMEM_VENDOR_MAJOR_VERSION`

Provides the major version of NVSHMEM as an Integer.

`NVSHMEM_VENDOR_MINOR_VERSION`

Provides the minor version of NVSHMEM as an Integer.

`NVSHMEM_VENDOR_PATCH_VERSION`

Provides the patch version of NVSHMEM as an Integer.

`NVSHMEM_VENDOR_VERSION`

A combination of `NVSHMEM_VENDOR_MAJOR_VERSION` `NVSHMEM_VENDOR_MINOR_VERSION` and `NVSHMEM_VENDOR_PATCH_VERSION` stylized as major * 10000 + minor * 100 + patch. NVSHMEM version 2.1.0 would report 20100.

`NVSHMEM_VENDOR_STRING`

A combination of `NVSHMEM_VENDOR_MAJOR_VERSION` `NVSHMEM_VENDOR_MINOR_VERSION` and `NVSHMEM_VENDOR_PATCH_VERSION` in string format stylized as “NVSHMEM v{major}.{minor}.patch.” NVSHMEM version 2.1.0 would report “NVSHMEM v2.1.0” In CorCpp{}, the string is terminated by a null character.
