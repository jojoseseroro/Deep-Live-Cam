#!/usr/bin/env python3

# Import the tkinter fix to patch the ScreenChanged error (module patches Tk on import)
import tkinter_fix  # noqa: F401

# Import standalone defaults and hardware probe
import modules.standalone_apply  # applies presets and sets globals defaults

import core

if __name__ == '__main__':
    core.run()
