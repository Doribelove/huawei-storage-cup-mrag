"""Released build filename; r58 qualified local AGX backend (not vLLM)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapter import build, environment
if __name__ == '__main__':
    build(environment())
