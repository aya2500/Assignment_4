import os
import torch
import urllib.request
from pathlib import Path
import tempfile
from urllib.parse import urlparse, unquote
import hashlib


def _cache_path_for_url(url: str) -> str:
    """Return a cache file path for a given URL.

    Preference order for cache location:
      1. env_settings().network_path if available
      2. ~/.cache/seqtrack_checkpoints/
    The filename is taken from the URL path's basename when possible, otherwise a hash is used.
    """
    # Try to get network_path from env settings (if user configured local env)
    try:
        from lib.test.evaluation.environment import env_settings

        env = env_settings()
        cache_dir = getattr(env, 'network_path', None)
    except Exception:
        cache_dir = None

    if not cache_dir:
        cache_dir = os.path.join(str(Path.home()), '.cache', 'seqtrack_checkpoints')

    os.makedirs(cache_dir, exist_ok=True)

    parsed = urlparse(url)
    basename = os.path.basename(parsed.path)
    basename = unquote(basename)
    if not basename:
        # fallback to hash-based name
        h = hashlib.sha1(url.encode('utf-8')).hexdigest()
        basename = f'checkpoint_{h}.pth.tar'

    return os.path.join(cache_dir, basename)

def load_checkpoint_from_url(url):
    """Load a checkpoint from a URL."""
    # Use a persistent cache so we don't re-download the same checkpoint every run.
    cache_file = _cache_path_for_url(url)
    if os.path.exists(cache_file):
        print(f"Using cached checkpoint: {cache_file}")
    else:
        print(f"Downloading checkpoint from {url} to {cache_file}")
        try:
            urllib.request.urlretrieve(url, cache_file)
        except Exception as e:
            # If download fails, ensure no partial file remains
            if os.path.exists(cache_file):
                try:
                    os.remove(cache_file)
                except Exception:
                    pass
            raise

    # Load the checkpoint onto GPU if available, otherwise CPU.
    if torch.cuda.is_available():
        checkpoint = torch.load(cache_file, map_location=lambda storage, loc: storage.cuda())
    else:
        checkpoint = torch.load(cache_file, map_location='cpu')

    return checkpoint


def load_checkpoint(path):
    """Load a local checkpoint onto GPU if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.load(path, map_location=lambda storage, loc: storage.cuda())
    else:
        return torch.load(path, map_location='cpu')