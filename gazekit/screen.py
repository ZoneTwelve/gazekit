"""Screen size detection without extra dependencies (macOS: osascript)."""

import subprocess
import sys

_FALLBACK = (1440, 900)


def screen_size() -> tuple[int, int]:
    """Return (width, height) of the main display in points."""
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["osascript", "-e",
                 'tell application "Finder" to get bounds of window of desktop'],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            # "0, 0, 1512, 982"
            parts = [int(p.strip()) for p in out.split(",")]
            if len(parts) == 4 and parts[2] > 0 and parts[3] > 0:
                return parts[2], parts[3]
        except Exception:
            pass
    try:
        import tkinter
        root = tkinter.Tk()
        size = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
        return size
    except Exception:
        return _FALLBACK
