"""Embed the Advanced Face Controls panel into the main Tkinter GUI if present.

This module attempts to attach a collapsible advanced controls panel to the
existing application root window. It runs asynchronously to avoid blocking UI
startup. It uses the same config and sliders as the standalone GUI.
"""
from __future__ import annotations

import threading
import time
import tkinter as tk
from tkinter import ttk

from modules.advanced_controls_config import get_current, write_config, DEFAULTS


def _attach_panel_to_root(root):
    # Create a Toplevel anchored to the main window
    top = tk.Toplevel(root)
    top.title('Advanced Face Controls')
    top.geometry('420x680')
    top.transient(root)
    top.attributes('-topmost', False)

    canvas = tk.Canvas(top)
    scrollbar = ttk.Scrollbar(top, orient='vertical', command=canvas.yview)
    scrollable = ttk.Frame(canvas)

    scrollable.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
    canvas.create_window((0, 0), window=scrollable, anchor='nw')
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side='left', fill='both', expand=True)
    scrollbar.pack(side='right', fill='y')

    vals = get_current()

    def add_slider(parent, key, label, frm, to, resolution=1):
        frame = ttk.Frame(parent)
        frame.pack(fill='x', padx=8, pady=6)
        lbl = ttk.Label(frame, text=label)
        lbl.pack(side='left')
        var = tk.DoubleVar(value=vals.get(key, DEFAULTS.get(key, 0)))
        scl = ttk.Scale(frame, from_=frm, to=to, variable=var, orient='horizontal')
        scl.pack(side='right', fill='x', expand=True)

        def on_change(*_):
            vals[key] = var.get()
            write_config(vals)

        var.trace_add('write', lambda *args: on_change())
        return var

    # Add a subset of sliders for the embedded UI to keep it compact
    add_slider(scrollable, 'identity', 'Global Identity Strength', 0, 100)
    add_slider(scrollable, 'eyes', 'Eyes', 0, 100)
    add_slider(scrollable, 'mouth', 'Mouth / Lips', 0, 100)
    add_slider(scrollable, 'jaw', 'Jawline', 0, 100)
    add_slider(scrollable, 'mask_feathering', 'Mask feathering (px)', 0, 40)
    add_slider(scrollable, 'blending_strength', 'Blending strength', 0.0, 1.0)
    add_slider(scrollable, 'temporal_smoothing', 'Temporal smoothing', 0.0, 1.0)

    # Buttons: Reset, Save profile, Load profile
    btn_frame = ttk.Frame(scrollable)
    btn_frame.pack(fill='x', pady=8, padx=8)

    def reset_defaults():
        write_config(DEFAULTS)
        top.destroy()

    ttk.Button(btn_frame, text='Reset to Default', command=reset_defaults).pack(side='left', padx=4)
    ttk.Button(btn_frame, text='Close', command=top.destroy).pack(side='right', padx=4)


def _wait_for_root_and_attach(timeout: float = 10.0):
    import tkinter as tk
    start = time.time()
    while time.time() - start < timeout:
        try:
            root = tk._default_root
            if root is not None:
                # Attach panel
                root.after(100, lambda: _attach_panel_to_root(root))
                return
        except Exception:
            pass
        time.sleep(0.2)


def start_embed_thread():
    t = threading.Thread(target=_wait_for_root_and_attach, daemon=True)
    t.start()


# Auto-start embedding when module is imported
try:
    start_embed_thread()
except Exception:
    pass
