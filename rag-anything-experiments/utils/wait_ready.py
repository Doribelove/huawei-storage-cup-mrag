from pathlib import Path
import sys
import time
import urllib.request
import urllib.parse
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapter import environment
config = environment()
address = config.get('RAG_BINDING_HOST', 'http://127.0.0.1:8080/query')
parts = urllib.parse.urlsplit(address)
health = urllib.parse.urlunsplit((parts.scheme, parts.netloc, '/health', '', ''))
deadline = time.monotonic() + 120
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(health, timeout=2) as result:
            if result.status == 200:
                break
    except OSError:
        time.sleep(0.5)
else:
    raise SystemExit('RAG server did not become ready within120seconds')
