"""
Interactive episode control for teleop data collection.

Lets the operator abort the episode in progress from the keyboard, either discarding it and
moving on to a freshly sampled one, or discarding it and retrying the *same* one - same
layout, same style, same object placements - so a fumbled attempt can be redone without
losing a scene setup that was worth keeping.
"""

from copy import deepcopy

from robosuite.wrappers import Wrapper
from termcolor import colored

SKIP = "skip"
RETRY = "retry"


def _parse_key(name):
    """
    Turns a key name into something pynput can be compared against.

    Args:
        name (str): key name, either a single character ("n") or a pynput special key
            ("f9", "esc", "tab", ...)

    Returns:
        pynput key object
    """
    from pynput import keyboard

    name = name.strip().lower()
    if hasattr(keyboard.Key, name):
        return getattr(keyboard.Key, name)
    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)
    raise ValueError(
        "unrecognized key '{}': expected a single character or a pynput key name "
        "such as 'f9', 'esc' or 'tab'".format(name)
    )


def _matches(key, target):
    """
    Whether a pressed key is the target key, comparing characters case-insensitively.

    Args:
        key: key reported by pynput
        target: key to compare against

    Returns:
        bool: True on a match
    """
    from pynput import keyboard

    if isinstance(target, keyboard.KeyCode):
        return (
            isinstance(key, keyboard.KeyCode)
            and key.char is not None
            and key.char.lower() == target.char
        )
    return key == target


class EpisodeKeyListener:
    """
    Background keyboard listener that records a request to abort the current episode.

    The listener is global rather than tied to the render window, so the keys work while the
    operator is in a headset and the simulator window does not have focus. The flip side is
    that presses also reach whatever window does have focus, so prefer keys that are not
    bound to anything else - the defaults (F9 / F10) are not used by MuJoCo's viewer.

    Args:
        skip_key (str): key that discards the episode and samples a new one

        retry_key (str): key that discards the episode and repeats the same one
    """

    def __init__(self, skip_key="f9", retry_key="f10"):
        self.skip_key_name = skip_key
        self.retry_key_name = retry_key
        self._skip_key = _parse_key(skip_key)
        self._retry_key = _parse_key(retry_key)
        self._request = None
        self._listener = None

    def start(self):
        """
        Starts listening. Failure to attach (eg no display on a headless box) is reported but
        not raised: losing the shortcut keys should not take data collection down with it.

        Returns:
            bool: True if the listener started
        """
        from pynput import keyboard

        try:
            self._listener = keyboard.Listener(on_press=self._on_press)
            # daemon so an unstopped listener cannot keep the process alive on exit
            self._listener.daemon = True
            self._listener.start()
        except Exception as e:
            print(
                colored(
                    "Could not start the episode key listener ({}: {}). Skip/retry keys "
                    "are disabled.".format(type(e).__name__, e),
                    "yellow",
                )
            )
            self._listener = None
            return False

        print(
            colored(
                "Episode keys: '{}' = discard and sample a new episode, "
                "'{}' = discard and retry this same episode".format(
                    self.skip_key_name, self.retry_key_name
                ),
                "green",
            )
        )
        return True

    def stop(self):
        """Stops listening. Teardown errors are swallowed - by the time this is called the
        backend connection may already be gone, and that is not worth an exception."""
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None

    def _on_press(self, key):
        if _matches(key, self._skip_key):
            self._request = SKIP
        elif _matches(key, self._retry_key):
            self._request = RETRY

    def clear(self):
        """Drops any pending request, so a stale press cannot abort the next episode."""
        self._request = None

    def take_request(self):
        """
        Reads and consumes the pending request.

        Returns:
            str or None: SKIP, RETRY, or None if no key was pressed
        """
        request, self._request = self._request, None
        return request


class EpisodeRepeatWrapper(Wrapper):
    """
    Allows the next reset to rebuild the episode that was just attempted.

    RoboCasa draws the layout, the style, which object instances are used and where they are
    placed from the env's numpy Generator, so restoring that generator's state before a reset
    reproduces the whole scene exactly.

    Pinning the recorded ep meta is not enough on its own. That fixes the layout, the style
    and the object configs, but the placement sampler still re-draws the pose, so the object
    lands somewhere else - which is the one thing a retry has to preserve.
    """

    def __init__(self, env):
        super().__init__(env)
        self._episode_rng_state = None
        self._pending_rng_state = None

    def repeat_next_episode(self):
        """
        Asks for the next reset to rebuild the episode that is currently loaded.
        """
        if self._episode_rng_state is not None:
            self._pending_rng_state = deepcopy(self._episode_rng_state)

    def reset(self, *args, **kwargs):
        rng = self.unwrapped.rng
        if self._pending_rng_state is not None:
            rng.bit_generator.state = self._pending_rng_state
            self._pending_rng_state = None
        # remember the state this episode is about to be built from, in case it gets retried
        self._episode_rng_state = deepcopy(rng.bit_generator.state)
        return self.env.reset(*args, **kwargs)
