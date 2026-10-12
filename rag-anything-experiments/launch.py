"""Import isolated locked client dependencies, then run the selected script."""
from pathlib import Path
import runpy
import sys
project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project/'vendor'))
target = Path(sys.argv[1]).resolve()
if not target.is_relative_to(project/'rag-anything-experiments'):
    raise ValueError('Entrypoint must belong to this release')
sys.argv = sys.argv[1:]
if target.name == 'judger.py':
    # The shipped demo has question-only inputs. Never grade it against empty
    # invented labels; the evaluator supplies its actual benchmark answers.
    from dotenv import load_dotenv
    import os
    import json
    load_dotenv(os.environ.get('DOTENV_FILE', str(project/'.env')), override=True)
    arguments = sys.argv[1:]
    input_name = arguments[arguments.index('--input_path')+1] if '--input_path' in arguments else os.getenv('BENCHMARK_NAME', 'UniDoc-Bench')+'_results/rag_answers.json'
    samples = json.loads(Path(input_name).read_text())
    if not samples or any(not isinstance(s.get('answer'), str) or not s['answer'].strip() for s in samples):
        raise ValueError('Judging requires evaluator-supplied gold answers. The packaged question-only demo has none; use run_rag_unidocbench to test inference.')
runpy.run_path(str(target), run_name='__main__')
