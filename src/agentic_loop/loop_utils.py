import os
import shutil
import hashlib, base64, pathlib
from utils import pjoin


# def get_file_hash(file_path):
#     """Generate a hash for a file to use as a unique identifier"""
#     with open(file_path, 'rb') as f:
#         file_hash = hashlib.md5(f.read()).hexdigest()
#     return file_hash


def get_file_name_hash(path, digest_bytes=6, prefix=""):
    """
    Return  `{stem}_{id}`  where `id` is a url-safe, base64 string of `digest_bytes` BLAKE2s output.
    • digest_bytes=6 → 48-bit hash → 8-char id (2**48 ≈ 2.8 x 10¹⁴ possibilities)
    """
    p = pathlib.Path(path)
    h  = hashlib.blake2s(digest_size=digest_bytes)
    h.update(p.read_bytes())
    # urlsafe_b64encode keeps it filename-safe; rstrip('=') removes padding
    short_id = base64.urlsafe_b64encode(h.digest()).decode('ascii').rstrip('=')
    return f"{prefix}{p.stem}_{short_id}"