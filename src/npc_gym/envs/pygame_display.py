"""Shared Pygame lifetime and presentation of independent renderer surfaces.

`pygame.quit()` tears down state that every renderer in the process shares, so a
renderer may only trigger it once no other environment is still drawing. Each
renderer, including a headless one, takes a share when it starts drawing and
returns it when it closes. Human frames share one window; their drawing surfaces
remain independent of that window's size and contents.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pygame import Surface

_open_displays = 0


def acquire() -> None:
    """Record that a renderer is using Pygame resources."""
    global _open_displays
    _open_displays += 1


def release() -> None:
    """Return a share and shut Pygame down once the last renderer has closed."""
    global _open_displays
    if _open_displays <= 0:
        return
    _open_displays -= 1
    if _open_displays == 0:
        import pygame

        pygame.display.quit()
        pygame.quit()


def open_displays() -> int:
    """Return how many renderers currently hold a share."""
    return _open_displays


def present(surface: "Surface", *, caption: str) -> None:
    """Show an independent frame in the shared window, resizing it as needed."""
    import pygame

    window = pygame.display.get_surface()
    if window is None or window.get_size() != surface.get_size():
        window = pygame.display.set_mode(surface.get_size())
    pygame.display.set_caption(caption)
    window.blit(surface, (0, 0))
    pygame.event.pump()
    pygame.display.flip()
