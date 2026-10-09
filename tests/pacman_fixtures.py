"""Original minimal geometry for terminal-transition tests."""

from npc_gym.envs.pacman.layout import PacmanLayout

TERMINAL_LAYOUT = PacmanLayout.from_text("%%%%%%%%%\n%P  G.. %\n%%%%%%%%%", name="terminal-fixture")
