"""Play the original mazes: hold arrows/WASD to turn, Space stops, P pauses, R restarts.

Install ``npc-gym[render]``. Choose --layout, --ghost-behavior, --seed and --fps.
The helper is intended for manual play and playability checks.
"""

import argparse
from contextlib import closing

from npc_gym.envs import PacmanEnv
from npc_gym.envs.pacman.layout import VECTORS

TURN_BUFFER_MS = 1000


def turn(env: PacmanEnv, current: int, requested: int) -> int:
    """Accept an open direction; a blocked request leaves movement unchanged."""
    state = env.state().game
    x, y = state.player.position
    dx, dy = VECTORS[requested]
    return current if (x + dx, y + dy) in state.layout.walls else requested


def main() -> None:
    import pygame

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=PacmanEnv.layouts, default="small")
    parser.add_argument(
        "--ghost-behavior", choices=["random", "deterministic", "partly-deterministic"], default="deterministic"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=5, help="moves per second (default: 5)")
    args = parser.parse_args()
    if args.fps < 1:
        parser.error("--fps must be positive")
    keys = {
        pygame.K_UP: 1,
        pygame.K_w: 1,
        pygame.K_DOWN: 2,
        pygame.K_s: 2,
        pygame.K_RIGHT: 3,
        pygame.K_d: 3,
        pygame.K_LEFT: 4,
        pygame.K_a: 4,
        pygame.K_SPACE: 0,
    }
    with closing(
        PacmanEnv(layout=args.layout, features="essential", ghost_behavior=args.ghost_behavior, render_mode="rgb_array")
    ) as env:
        env.reset(seed=args.seed)
        initial_frame = env.render()
        assert initial_frame is not None
        height, width = initial_frame.shape[:2]
        pygame.display.init()
        if pygame.display.get_driver() in {"dummy", "offscreen"}:
            parser.exit(
                1,
                f"Cannot show a window with SDL's {pygame.display.get_driver()!r} video driver. "
                "Run from a desktop terminal with X11/Wayland libraries available; "
                "unset SDL_VIDEODRIVER if it selects a headless driver. "
                "After updating shell.nix, exit and re-enter nix-shell.\n",
            )
        window = pygame.display.set_mode((width, height))
        pygame.display.set_caption("NPC Gym — arrows / WASD, Space stop, P pause, R restart")
        clock = pygame.time.Clock()
        action, paused, done, running = 0, True, False, True
        held: dict[int, int] = {}
        pending: tuple[int, int | None] | None = None  # Key and expiry; held keys do not expire.
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
                elif event.type == pygame.KEYUP:
                    held.pop(event.key, None)
                    if pending is not None and pending[0] == event.key:
                        pending = event.key, pygame.time.get_ticks() + TURN_BUFFER_MS
                elif event.type == pygame.WINDOWFOCUSLOST:
                    held.clear()
                    pending = None
                    action, paused = 0, True
                elif event.type == pygame.KEYDOWN:
                    if event.key in keys:
                        if event.key == pygame.K_SPACE:
                            held.clear()
                        held.pop(event.key, None)
                        held[event.key] = keys[event.key]
                        pending = event.key, None
                        paused = False
                    elif event.key == pygame.K_p:
                        paused = not paused
                        pending = None
                    elif event.key == pygame.K_r:
                        env.reset(seed=args.seed)
                        held.clear()
                        pending = None
                        done, paused, action = False, True, 0
            if running and not paused and not done:
                if pending is not None and pending[1] is not None and pygame.time.get_ticks() > pending[1]:
                    pending = None
                requested = keys[pending[0]] if pending is not None else next(reversed(held.values()), action)
                action = turn(env, action, requested)
                if pending is not None and action == requested:
                    pending = None
                _, _, done, _, _ = env.step(action)
            frame = env.render()
            assert frame is not None
            pygame.surfarray.blit_array(window, frame.transpose(1, 0, 2))
            pygame.display.flip()
            clock.tick(args.fps)


if __name__ == "__main__":
    main()
