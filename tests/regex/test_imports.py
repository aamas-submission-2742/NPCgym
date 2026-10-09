import os
import subprocess
import sys

SCRIPT = """
import importlib.abc
import sys
class BlockCompiler(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'ltlf2dfa', 'lark', 'sympy'} or fullname.startswith('npc_gym.monitors.ltlf'):
            raise AssertionError('Unexpected compiler import: ' + fullname)
sys.meta_path.insert(0, BlockCompiler())
from npc_gym.monitors.regex import compile_regex, from_regex
from npc_gym.monitors import MonitorInput
monitor = from_regex('.*[a]', propositions={'a': lambda i: 'a' in i.labels}, reporting='prefix')
assert monitor.reset(MonitorInput(frozenset({'a'}))) is True
definition = compile_regex('([a] | [b & !c])*[c]?')
print(repr((definition.atoms, definition.states, sorted(definition.final_states), dict(definition.transitions))))
"""


def test_no_optional_compiler_and_deterministic_numbering_across_hash_seeds():
    results = [
        subprocess.run(
            [sys.executable, "-c", SCRIPT],
            env={**os.environ, "PYTHONHASHSEED": seed},
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for seed in ("1", "42")
    ]
    assert results[0] == results[1]
