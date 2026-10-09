import subprocess
import sys
from pathlib import Path

import npc_gym

CORE_IMPORT_SCRIPT = """
import importlib
import importlib.abc
import sys

class BlockCompiler(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'ltlf2dfa', 'lark', 'sympy'}:
            raise AssertionError('Unexpected optional import: ' + fullname)
sys.meta_path.insert(0, BlockCompiler())
import npc_gym
from npc_gym.automata import CompiledDFA
from npc_gym.monitors import AutomatonMonitor, MonitorInput
assert not any(name.startswith('npc_gym.monitors.ltlf') for name in sys.modules)
definition = CompiledDFA((), 0, frozenset({0}), {0: [("", 0)]})
monitor = AutomatonMonitor(definition, propositions={}, reporting="prefix")
assert monitor.reset(MonitorInput(frozenset())) is True
assert monitor.update(MonitorInput(frozenset())) is True
import npc_gym.monitors.ltlf
for domain in ('merchant', 'gardener', 'pacman', 'taxi'):
    module = importlib.import_module('npc_gym.monitors.' + domain + '_monitors')
    registry = getattr(module, domain + '_monitor_registry')
    for identifier in registry.ids:
        monitor = registry.make(identifier)
        monitor.reset(MonitorInput(frozenset()))
        monitor.update(MonitorInput(frozenset(), truncated=True))
        assert monitor.count >= 0 if hasattr(monitor, "count") else all(v >= 0 for v in monitor.counts.values())
"""


def test_core_import_and_every_regex_builtin_without_optional_compiler():
    source = str(Path(npc_gym.__file__).resolve().parent.parent)
    subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.insert(0, " + repr(source) + ")\n" + CORE_IMPORT_SCRIPT],
        check=True,
        capture_output=True,
        text=True,
    )
