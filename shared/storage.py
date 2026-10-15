"""Small object-storage interface for local development and Google Cloud Storage."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Optional, Protocol


class ObjectStorage(Protocol):
    def put_bytes(self, key: str, data: bytes, content_type: str) -> None: ...
    def get_bytes(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def uri(self, key: str) -> Optional[str]: ...


def _safe_key(key: str) -> str:
    path = PurePosixPath(key)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError("Storage keys must be relative paths without '.' or '..' segments")
    return path.as_posix()


class LocalObjectStorage:
    """Filesystem implementation used when no GCS bucket is configured."""

    def __init__(self, root: str | Path | None = None):
        default_root = Path(__file__).resolve().parents[1] / "storage_data"
        self.root = Path(root or os.getenv("LOCAL_STORAGE_DIR", default_root)).resolve()

    def _path(self, key: str) -> Path:
        path = (self.root / _safe_key(key)).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Storage key escapes the configured storage directory")
        return path

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        del content_type
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def uri(self, key: str) -> Optional[str]:
        del key
        return None


class GCSObjectStorage:
    """GCS implementation authenticated with Application Default Credentials."""

    def __init__(self, bucket_name: str):
        from google.cloud import storage

        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.bucket_name = bucket_name

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        blob = self.bucket.blob(_safe_key(key))
        blob.upload_from_string(data, content_type=content_type)

    def get_bytes(self, key: str) -> bytes:
        return self.bucket.blob(_safe_key(key)).download_as_bytes()

    def exists(self, key: str) -> bool:
        return self.bucket.blob(_safe_key(key)).exists()

    def uri(self, key: str) -> str:
        return f"gs://{self.bucket_name}/{_safe_key(key)}"


def get_storage() -> ObjectStorage:
    """Select GCS when GCS_BUCKET is configured; otherwise use local storage."""

    bucket_name = os.getenv("GCS_BUCKET")
    if bucket_name:
        return GCSObjectStorage(bucket_name)
    return LocalObjectStorage()
