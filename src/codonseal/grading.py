"""HumanEval checks in an OS sandbox, separate from decoding and selection."""
import json
import signal
from pathlib import Path
import subprocess
import sys
import tempfile


def grade(row,text):
    if sys.platform!='darwin':
        raise RuntimeError('HumanEval execution requires the provided macOS sandbox; use a separately isolated grader on other systems.')
    root=Path(__file__).resolve().parent
    command=['/usr/bin/sandbox-exec','-f',str(root/'humaneval.sb'),'-D',f'worker={root/"worker.py"}',
             '-D',f'runtime={sys.base_prefix}','-D',f'venv={sys.prefix}',sys.executable,'-I','-B',str(root/'worker.py')]
    with tempfile.TemporaryDirectory(prefix='codonseal-grade-') as scratch:
        try:
            result=subprocess.run(command,input=json.dumps(dict(text=text,test=row['test'],entry_point=row['entry_point'])),
                capture_output=True,text=True,timeout=5,cwd=scratch,env={'PATH':'/usr/bin:/bin'})
        except subprocess.TimeoutExpired:return 'timeout'
    if 'sandbox-exec:' in result.stderr:
        raise RuntimeError('Grading isolation failed: '+result.stderr[:500])
    if result.returncode == -signal.SIGXCPU:
        return 'timeout'
    if result.returncode!=0:
        raise RuntimeError(f'Grading worker exited ({result.returncode}): '+result.stderr[:500])
    return result.stdout.strip()
