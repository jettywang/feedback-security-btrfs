from __future__ import annotations


class Error(Exception):
    pass


class InconsistentFilesystemError(Error, ValueError):
    """The supplied images cannot be shown to belong to one readable filesystem state."""


class ConflictingDeviceError(InconsistentFilesystemError):
    """A devid is claimed by more than one device identity."""


class ConflictingChunkError(InconsistentFilesystemError):
    """Two chunk items map the same logical range differently."""


class InvalidSuperblockError(InconsistentFilesystemError):
    """A superblock failed magic, placement or checksum validation."""


class InvalidTreeBlockError(Error):
    """A tree block failed checksum, address, ownership or generation validation."""


class MissingDeviceError(Error):
    """A chunk stripe references a device that was not supplied."""


class UnsupportedChecksumError(Error):
    """A tree block uses a checksum algorithm this parser cannot verify."""


class FileNotFoundError(Error, FileNotFoundError):
    pass


class IsADirectoryError(Error, IsADirectoryError):
    pass


class NotADirectoryError(Error, NotADirectoryError):
    pass


class NotASymlinkError(Error):
    pass
