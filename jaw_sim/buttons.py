"""The four hardware buttons (MK424Pro macro pad, connected by Bluetooth).

The pad types the letters a, b, c, d. This listener watches the keyboard system-wide (pynput), so
the buttons work no matter which window is in front, and puts the matching command name into a
queue that the simulation drains every frame. No simulator code in here: the same listener will
feed Isaac Sim later.

macOS needs "Input Monitoring" permission for the terminal that runs the sim
(System Settings > Privacy & Security > Input Monitoring).
"""

import queue

from pynput import keyboard

BUTTONS = {
    "a": "calibrate",  # start calibration / record the next reference point
    "b": "operate",  # operating mode: tooth commands are executed
    "c": "stop",  # cancel the current move and hold position; leaves operating mode
    "d": "park",  # pull the ejector out of the mouth to the park pose
}


class ButtonListener:
    """Turns global key presses of the BUTTONS letters into command names on a queue."""

    def __init__(self, commands: queue.Queue):
        self.commands = commands
        self.listener = keyboard.Listener(on_press=self._on_press)

    def start(self) -> None:
        self.listener.start()

    def stop(self) -> None:
        self.listener.stop()

    def _on_press(self, key) -> None:
        letter = getattr(key, "char", None)
        if letter in BUTTONS:
            self.commands.put(BUTTONS[letter])
