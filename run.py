#!/usr/bin/env python3

# Import the tkinter fix to patch the ScreenChanged error (module patches Tk on import)
import tkinter_fix  # noqa: F401

# Import standalone defaults and hardware probe
import modules.standalone_apply  # applies presets and sets globals defaults

# Import the face_swapper injector so advanced blending is applied (monkey-patch)
try:
    import modules.face_swapper_inject  # noqa: F401
except Exception:
    pass

# Import embedded advanced controls UI so it attaches to the main window
try:
    import modules.advanced_controls_embed  # noqa: F401
except Exception:
    pass

import core

if __name__ == '__main__':
    core.run()
