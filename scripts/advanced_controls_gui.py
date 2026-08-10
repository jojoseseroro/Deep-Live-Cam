"""Simple Tkinter-based GUI for adjusting Advanced Face Controls.

This standalone GUI writes the advanced controls JSON to the user's home
as .deeplivecam_advanced_controls.json. The main DeepLiveCam process reads
this file (via modules.advanced_controls_config) and updates the live blending.

Run this separately while the main app is running to control live parameters.
"""

import json
import os
import tkinter as tk
from tkinter import ttk

from modules.advanced_controls_config import CONFIG_PATH, DEFAULTS, write_config


class AdvancedControlsApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DeepLiveCam — Advanced Face Controls")
        self.geometry("520x720")
        self.resizable(False, True)

        self.values = dict(DEFAULTS)

        self.create_widgets()

    def create_widgets(self):
        # Scrollable frame
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable = ttk.Frame(canvas)

        scrollable.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Add sliders for region strengths
        def add_slider(parent, key, label, from_, to, row=None):
            frame = ttk.Frame(parent)
            frame.pack(fill="x", padx=8, pady=6)
            lbl = ttk.Label(frame, text=label)
            lbl.pack(side="left")
            var = tk.DoubleVar(value=self.values.get(key, DEFAULTS.get(key, 0)))
            scl = ttk.Scale(frame, from_=from_, to=to, variable=var, orient="horizontal")
            scl.pack(side="right", fill="x", expand=True)

            def on_change(event=None):
                self.values[key] = var.get()
                write_config(self.values)

            var.trace_add("write", lambda *args: on_change())
            return var

        # Identity and regions
        add_slider(scrollable, "identity", "Global Identity Strength", 0, 100)
        add_slider(scrollable, "forehead", "Forehead", 0, 100)
        add_slider(scrollable, "eyebrows", "Eyebrows", 0, 100)
        add_slider(scrollable, "eyes", "Eyes", 0, 100)
        add_slider(scrollable, "upper_eyelids", "Upper eyelids", 0, 100)
        add_slider(scrollable, "lower_eyelids", "Lower eyelids", 0, 100)
        add_slider(scrollable, "nose", "Nose", 0, 100)
        add_slider(scrollable, "cheeks", "Cheeks", 0, 100)
        add_slider(scrollable, "cheekbones", "Cheekbones", 0, 100)
        add_slider(scrollable, "mouth", "Mouth / Lips", 0, 100)
        add_slider(scrollable, "jaw", "Jawline", 0, 100)
        add_slider(scrollable, "chin", "Chin", 0, 100)
        add_slider(scrollable, "face_shape", "Face shape / contour", 0, 100)
        add_slider(scrollable, "skin_texture", "Skin / Texture", 0, 100)
        add_slider(scrollable, "color_matching", "Color / Skin-tone matching", 0, 100)
        add_slider(scrollable, "hairline", "Hairline / Forehead boundary", 0, 100)

        ttk.Separator(scrollable).pack(fill="x", pady=6)

        # Preservation controls
        add_slider(scrollable, "expression_preservation", "Expression preservation", 0, 100)
        add_slider(scrollable, "eye_blink_preservation", "Eye / Blink preservation", 0, 100)
        add_slider(scrollable, "mouth_movement_preservation", "Mouth movement preservation", 0, 100)
        add_slider(scrollable, "teeth_preservation", "Teeth / Mouth preservation", 0, 100)

        ttk.Separator(scrollable).pack(fill="x", pady=6)

        # Mask & blending controls
        add_slider(scrollable, "mask_size", "Mask size (percent)", 50, 150)
        add_slider(scrollable, "mask_feathering", "Mask feathering (px)", 0, 40)
        add_slider(scrollable, "blending_strength", "Blending strength", 0.0, 1.0)
        add_slider(scrollable, "color_correction_strength", "Color correction strength", 0.0, 1.0)
        add_slider(scrollable, "enhancer_strength", "Enhancer strength", 0.0, 1.0)
        add_slider(scrollable, "temporal_smoothing", "Temporal smoothing", 0.0, 1.0)

        ttk.Separator(scrollable).pack(fill="x", pady=6)

        # Toggle
        var_enabled = tk.BooleanVar(value=True)
        chk = ttk.Checkbutton(scrollable, text="Advanced controls enabled", variable=var_enabled)
        chk.pack(anchor="w", padx=8, pady=8)

        def on_enable():
            self.values["advanced_enabled"] = var_enabled.get()
            write_config(self.values)

        var_enabled.trace_add("write", lambda *args: on_enable())

        # Save / Load buttons
        btns = ttk.Frame(scrollable)
        btns.pack(fill="x", pady=8, padx=8)
        def save_now():
            write_config(self.values)

        ttk.Button(btns, text="Save", command=save_now).pack(side="left", padx=4)
        ttk.Button(btns, text="Reset", command=self.reset_defaults).pack(side="right", padx=4)

    def reset_defaults(self):
        self.values = dict(DEFAULTS)
        write_config(self.values)
        self.destroy()
        # Relaunch to pick up defaults
        self.__init__()


if __name__ == "__main__":
    app = AdvancedControlsApp()
    # Ensure default config exists
    if not os.path.exists(CONFIG_PATH):
        write_config(DEFAULTS)
    app.mainloop()
