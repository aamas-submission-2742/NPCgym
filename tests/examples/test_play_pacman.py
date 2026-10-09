"""Headless smoke for keyboard controls and window cleanup."""

import pygame
import pytest

from examples import play_pacman
from npc_gym.envs import PacmanEnv, PacmanLayout, pygame_display


@pytest.fixture
def play(monkeypatch):
    """Drive real transitions through keyboard events on a fixed T junction."""
    actions, positions = [], []
    layout = PacmanLayout.from_text("%%%%%%%%%\n%%%.....%\n%%%.%%%%%\n%P......%\n%%%%%%%%%")

    class RecordingGame(PacmanEnv):
        def __init__(self, **kwargs):
            super().__init__(**{**kwargs, "layout": layout})

        def step(self, action):
            result = super().step(action)
            actions.append(action)
            positions.append(self.state().game.player.position)
            return result

    monkeypatch.setattr(play_pacman, "PacmanEnv", RecordingGame)
    monkeypatch.setattr("sys.argv", ["play_pacman", "--fps", "1000"])
    monkeypatch.setattr(pygame.display, "get_driver", lambda: "x11")

    def run(frames, *, times=None, maze=None):
        nonlocal layout
        if maze is not None:
            layout = PacmanLayout.from_text(maze)
        now = 0
        frames = iter(enumerate([*frames, [pygame.event.Event(pygame.QUIT)]]))

        def events():
            nonlocal now
            index, frame = next(frames)
            now = index * 200 if times is None else times[min(index, len(times) - 1)]
            return frame

        monkeypatch.setattr(pygame.event, "get", events)
        monkeypatch.setattr(pygame.time, "get_ticks", lambda: now)
        play_pacman.main()
        assert pygame_display.open_displays() == 0
        return actions, positions

    return run


def press(key):
    return pygame.event.Event(pygame.KEYDOWN, key=key)


def release(key):
    return pygame.event.Event(pygame.KEYUP, key=key)


@pytest.mark.parametrize(("right", "up"), [(pygame.K_RIGHT, pygame.K_UP), (pygame.K_d, pygame.K_w)])
def test_blocked_held_turn_keeps_moving_then_turns_at_the_junction(play, right, up):
    actions, positions = play([[press(right)], [press(up)], []])
    assert actions == [3, 3, 1]
    assert positions == [(2, 1), (3, 1), (3, 2)]


def test_released_turn_is_buffered_until_the_next_opening(play):
    actions, positions = play([[press(pygame.K_RIGHT)], [press(pygame.K_UP)], [release(pygame.K_UP)]])
    assert actions == [3, 3, 1]
    assert positions == [(2, 1), (3, 1), (3, 2)]


@pytest.mark.parametrize(("elapsed", "last_action"), [(1000, 1), (1001, 3)])
def test_released_turn_expires_after_one_second_of_real_time(play, elapsed, last_action):
    actions, _ = play(
        [[press(pygame.K_RIGHT)], [press(pygame.K_UP), release(pygame.K_UP)], []],
        times=[0, 200, 200 + elapsed],
    )
    assert actions == [3, 3, last_action]


@pytest.mark.parametrize("replacement", [pygame.K_RIGHT, pygame.K_DOWN])
def test_new_direction_replaces_buffer_even_when_the_new_direction_is_blocked(play, replacement):
    actions, _ = play(
        [
            [press(pygame.K_RIGHT)],
            [press(pygame.K_UP), release(pygame.K_UP)],
            [press(replacement), release(replacement)],
            [],
        ]
    )
    assert actions == [3, 3, 3, 3]


def test_held_direction_does_not_expire(play):
    actions, _ = play([[press(pygame.K_RIGHT)], [press(pygame.K_UP)], []], times=[0, 200, 1201])
    assert actions == [3, 3, 1]


@pytest.mark.parametrize(("since_release", "last_action"), [(1000, 1), (1001, 3)])
def test_long_hold_gets_a_full_buffer_interval_after_release(play, since_release, last_action):
    actions, positions = play(
        [[press(pygame.K_RIGHT)], [press(pygame.K_UP)], [release(pygame.K_UP)], []],
        times=[0, 100, 2100, 2100 + since_release],
        maze="%%%%%%%%%\n%%%%....%\n%%%%.%%%%\n%P......%\n%%%%%%%%%",
    )
    assert actions == [3, 3, 3, last_action]
    assert positions[-1] == ((4, 2) if last_action == 1 else (5, 1))


def test_releasing_an_older_key_does_not_replace_the_new_buffer(play):
    actions, _ = play([[press(pygame.K_RIGHT)], [press(pygame.K_UP)], [press(pygame.K_DOWN), release(pygame.K_UP)], []])
    assert actions == [3, 3, 3, 3]


def test_pausing_clears_buffered_turn(play):
    actions, _ = play(
        [
            [press(pygame.K_RIGHT)],
            [press(pygame.K_UP)],
            [press(pygame.K_p)],
            [release(pygame.K_UP)],
            [press(pygame.K_p)],
        ],
        times=[0, 100, 200, 250, 300],
    )
    assert actions == [3, 3, 3]


def test_short_tap_turns_when_open_even_if_released_before_the_frame(play):
    actions, positions = play(
        [[press(pygame.K_RIGHT), release(pygame.K_RIGHT)], [], [press(pygame.K_UP), release(pygame.K_UP)], []]
    )
    assert actions == [3, 3, 1, 1]
    assert positions[-1] == (3, 3)


def test_most_recent_held_key_takes_priority_then_older_key_resumes(play):
    actions, positions = play([[press(pygame.K_RIGHT)], [press(pygame.K_UP)], [], [release(pygame.K_UP)], []])
    assert actions == [3, 3, 1, 1, 3]
    assert positions[-1] == (4, 3)


def test_space_stops_and_clears_held_turns(play):
    actions, positions = play(
        [[press(pygame.K_RIGHT)], [press(pygame.K_UP)], [press(pygame.K_SPACE)], [release(pygame.K_SPACE)]]
    )
    assert actions == [3, 3, 0, 0]
    assert positions[-1] == (3, 1)


@pytest.mark.parametrize("interrupt", [press(pygame.K_r), pygame.event.Event(pygame.WINDOWFOCUSLOST)])
def test_restart_and_focus_loss_clear_pending_turns_and_pause(play, interrupt):
    actions, _ = play(
        [[press(pygame.K_RIGHT)], [press(pygame.K_UP)], [interrupt], [], [press(pygame.K_p)]],
        times=[0, 50, 100, 150, 200],
    )
    assert actions == [3, 3, 0]


def test_keyboard_helper_handles_move_pause_restart_and_quit(monkeypatch):
    monkeypatch.setattr("sys.argv", ["play_pacman", "--layout", "small", "--fps", "1000"])
    # Exercise keyboard handling on the dummy display without claiming a real window.
    monkeypatch.setattr(pygame.display, "get_driver", lambda: "x11")
    events = iter(
        [
            [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT)],
            [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p)],
            [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r)],
            [pygame.event.Event(pygame.QUIT)],
        ]
    )
    monkeypatch.setattr(pygame.event, "get", lambda: next(events))
    play_pacman.main()
    assert pygame_display.open_displays() == 0
    assert pygame.display.get_surface() is None


@pytest.mark.parametrize("driver", ["dummy", "offscreen"])
def test_keyboard_helper_rejects_invisible_display(monkeypatch, capsys, driver):
    monkeypatch.setattr("sys.argv", ["play_pacman"])
    monkeypatch.setattr(pygame.display, "get_driver", lambda: driver)
    with pytest.raises(SystemExit) as error:
        play_pacman.main()
    assert error.value.code == 1
    message = capsys.readouterr().err
    assert driver in message
    assert "Cannot show a window" in message
    assert "SDL_VIDEODRIVER" in message
    assert "nix-shell" in message
    assert pygame_display.open_displays() == 0
    assert pygame.display.get_surface() is None
