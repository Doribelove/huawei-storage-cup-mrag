"""Released server filename; serial r58 AGX HTTP interface, not a vLLM claim."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapter import environment, serve
if __name__ == '__main__':
    serve(environment())
