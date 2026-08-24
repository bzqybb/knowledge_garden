from __future__ import annotations

import argparse
import base64
import ctypes
import getpass
import os
from ctypes import wintypes
from pathlib import Path


MAGIC = b"KGDPAPI2\n"
ENTROPY = b"KnowledgeGarden.DPAPI.v2"
CRYPTPROTECT_UI_FORBIDDEN = 0x1


class CredentialError(RuntimeError):
    pass


class DataBlob(ctypes.Structure):
    _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]


def _windows_apis():
    if os.name != "nt":
        raise CredentialError("Windows DPAPI is only available on Windows.")
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    crypt32.CryptProtectData.argtypes = [
        ctypes.POINTER(DataBlob),
        wintypes.LPCWSTR,
        ctypes.POINTER(DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(DataBlob),
    ]
    crypt32.CryptProtectData.restype = wintypes.BOOL
    crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(DataBlob),
        ctypes.c_void_p,
        ctypes.POINTER(DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(DataBlob),
    ]
    crypt32.CryptUnprotectData.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    return crypt32, kernel32


def _blob_from_bytearray(value: bytearray) -> tuple[DataBlob, object]:
    if value:
        buffer = (ctypes.c_ubyte * len(value)).from_buffer(value)
        return DataBlob(len(value), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))), buffer
    buffer = (ctypes.c_ubyte * 1)()
    return DataBlob(0, ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))), buffer


def _dpapi(value: bytearray, *, protect: bool) -> bytearray:
    crypt32, kernel32 = _windows_apis()
    entropy = bytearray(ENTROPY)
    input_blob, input_buffer = _blob_from_bytearray(value)
    entropy_blob, entropy_buffer = _blob_from_bytearray(entropy)
    output_blob = DataBlob()
    function = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    try:
        ok = function(
            ctypes.byref(input_blob),
            "Knowledge Garden API Key" if protect else None,
            ctypes.byref(entropy_blob),
            None,
            None,
            CRYPTPROTECT_UI_FORBIDDEN,
            ctypes.byref(output_blob),
        )
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())
        return bytearray(ctypes.string_at(output_blob.data, output_blob.size))
    except OSError as exc:
        raise CredentialError(f"Windows DPAPI failed: {exc}") from exc
    finally:
        _ = input_buffer, entropy_buffer
        value[:] = b"\0" * len(value)
        entropy[:] = b"\0" * len(entropy)
        if output_blob.data:
            ctypes.memset(output_blob.data, 0, output_blob.size)
            kernel32.LocalFree(output_blob.data)


def protect_secret(secret: str) -> bytes:
    if not secret:
        raise CredentialError("API Key cannot be empty.")
    plain = bytearray(secret.encode("utf-8"))
    encrypted = _dpapi(plain, protect=True)
    try:
        return MAGIC + base64.b64encode(encrypted)
    finally:
        encrypted[:] = b"\0" * len(encrypted)


def unprotect_secret(payload: bytes) -> str:
    if not payload.startswith(MAGIC):
        raise CredentialError("Unsupported credential format; save the API Key again.")
    try:
        encrypted = bytearray(base64.b64decode(payload[len(MAGIC) :], validate=True))
    except ValueError as exc:
        raise CredentialError("Saved credential is damaged; save the API Key again.") from exc
    plain = _dpapi(encrypted, protect=False)
    try:
        return plain.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CredentialError("Saved credential is damaged; save the API Key again.") from exc
    finally:
        plain[:] = b"\0" * len(plain)


def save_secret(path: Path, secret: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(protect_secret(secret))
    os.replace(temporary, path)


def load_secret(path: Path) -> str:
    return unprotect_secret(path.read_bytes()) if path.is_file() else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Knowledge Garden Windows credential helper")
    parser.add_argument("action", choices=("save", "test"))
    parser.add_argument("path", nargs="?", type=Path)
    parser.add_argument(
        "--prompt",
        default="Paste the DeepSeek API Key (input is hidden): ",
        help="Hidden-input prompt used by the save action.",
    )
    parser.add_argument(
        "--saved-label",
        default="API key",
        help="Non-secret label shown after a credential is saved.",
    )
    parser.add_argument(
        "--show-fingerprint",
        action="store_true",
        help="Show only the saved key prefix and length for verification.",
    )
    args = parser.parse_args()
    if args.action == "save":
        if args.path is None:
            parser.error("save requires a credential path")
        secret = getpass.getpass(args.prompt).strip()
        save_secret(args.path, secret)
        print(f"{args.saved_label} saved with Windows DPAPI for the current Windows user.")
        if args.show_fingerprint:
            print(f"Saved key fingerprint: prefix={secret[:5]!r}, length={len(secret)}")
        return
    sample = "sk-test-local-only"
    if unprotect_secret(protect_secret(sample)) != sample:
        raise CredentialError("Windows DPAPI self-test failed.")
    print("Windows DPAPI self-test: PASS")


if __name__ == "__main__":
    main()
