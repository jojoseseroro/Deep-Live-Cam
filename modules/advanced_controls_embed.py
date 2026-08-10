"""Embedded Advanced Face Controls panel for the main Tk GUI.

This module creates a dockable/collapsible advanced controls panel attached
to the main Tk root window. It provides the full set of sliders requested,
live mask preview, clickable region selection, Save/Load profile and Reset.

It writes changes to the same JSON config file used by the standalone editor:
~/.deeplivecam_advanced_controls.json

This implementation is conservative and depends on tkinter being used by the
main application. It polls the face_state to draw previews.
"""
from __future__ import annotations

import threading
import time
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
import numpy as np
import cv2

from modules.advanced_controls_config import get_current, write_config, DEFAULTS, CONFIG_PATH
from modules import advanced_face_controls as afc
from modules import face_state
from modules import profile_manager
from modules.standalone_flags import PRESETS

REGION_KEYS = [
    'forehead','eyebrows','eyes','upper_eyelids','lower_eyelids','nose',
    'cheeks','cheekbones','mouth','jaw','chin','face_shape','skin_texture','color_matching','hairline'
]

POLL_INTERVAL = 0.1


def _make_slider(parent, key, label, frm, to, resolution=1, is_float=False):
    frame = ttk.Frame(parent)
    frame.pack(fill='x', padx=6, pady=4)
    ttk.Label(frame, text=label).pack(side='left')
    var = tk.DoubleVar(value=get_current().get(key, DEFAULTS.get(key, 0)))
    scl = ttk.Scale(frame, from_=frm, to=to, variable=var, orient='horizontal')
    scl.pack(side='right', fill='x', expand=True)

    def on_change(*_):
        cur = get_current()
        cur[key] = float(var.get()) if is_float else float(int(var.get()))
        write_config(cur)

    var.trace_add('write', lambda *args: on_change())
    return var


class AdvancedPanel:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.top = tk.Toplevel(root)
        self.top.title('Advanced Face Controls')
        self.top.geometry('520x820')
        self.top.protocol('WM_DELETE_WINDOW', self.top.withdraw)

        # Left: canvas preview
        left = ttk.Frame(self.top)
        left.pack(side='left', fill='both', expand=True)
        self.canvas = tk.Canvas(left, bg='black', width=480, height=360)
        self.canvas.pack(fill='both', expand=True, padx=6, pady=6)
        self.canvas.bind('<Button-1>', self.on_canvas_click)

        # Right: controls
        right = ttk.Frame(self.top)
        right.pack(side='right', fill='y')

        # Add region sliders
        self.sliders = {}
        ttk.Label(right, text='Region Controls').pack(anchor='w', padx=6)
        for key in REGION_KEYS:
            self.sliders[key] = _make_slider(right, key, key.replace('_', ' ').title(), 0, 100)

        ttk.Separator(right).pack(fill='x', pady=6)
        ttk.Label(right, text='Preservation Controls').pack(anchor='w', padx=6)
        self.sliders['expression_preservation'] = _make_slider(right, 'expression_preservation', 'Expression preservation', 0, 100)
        self.sliders['eye_blink_preservation'] = _make_slider(right, 'eye_blink_preservation', 'Eye/Blink preservation', 0, 100)
        self.sliders['mouth_movement_preservation'] = _make_slider(right, 'mouth_movement_preservation', 'Mouth movement preservation', 0, 100)
        self.sliders['teeth_preservation'] = _make_slider(right, 'teeth_preservation', 'Teeth preservation', 0, 100)

        ttk.Separator(right).pack(fill='x', pady=6)
        ttk.Label(right, text='Mask & Blend').pack(anchor='w', padx=6)
        self.sliders['mask_size'] = _make_slider(right, 'mask_size', 'Mask size (%)', 50, 150)
        self.sliders['mask_feathering'] = _make_slider(right, 'mask_feathering', 'Mask feathering (px)', 0, 40)
        self.sliders['blending_strength'] = _make_slider(right, 'blending_strength', 'Blending strength', 0.0, 1.0, is_float=True)
        self.sliders['color_correction_strength'] = _make_slider(right, 'color_correction_strength', 'Color correction strength', 0.0, 1.0, is_float=True)
        self.sliders['enhancer_strength'] = _make_slider(right, 'enhancer_strength', 'Enhancer strength', 0.0, 1.0, is_float=True)
        self.sliders['temporal_smoothing'] = _make_slider(right, 'temporal_smoothing', 'Temporal smoothing', 0.0, 1.0, is_float=True)

        ttk.Separator(right).pack(fill='x', pady=6)
        ttk.Label(right, text='Global').pack(anchor='w', padx=6)
        self.sliders['identity'] = _make_slider(right, 'identity', 'Global identity strength', 0, 100)

        # Preset buttons
        btn_frame = ttk.Frame(right)
        btn_frame.pack(fill='x', pady=6, padx=6)
        ttk.Button(btn_frame, text='Natural', command=lambda: self.apply_preset('low_latency')).pack(fill='x')
        ttk.Button(btn_frame, text='Balanced', command=lambda: self.apply_preset('balanced')).pack(fill='x')
        ttk.Button(btn_frame, text='High Similarity', command=lambda: self.apply_preset('high_quality')).pack(fill='x')
        ttk.Button(btn_frame, text='Maximum Quality', command=lambda: self.apply_preset('maximum')).pack(fill='x')

        # Profile management
        prof_frame = ttk.Frame(right)
        prof_frame.pack(fill='x', pady=6, padx=6)
        ttk.Button(prof_frame, text='Save Profile', command=self.save_profile).pack(fill='x')
        ttk.Button(prof_frame, text='Load Profile', command=self.load_profile).pack(fill='x')
        ttk.Button(prof_frame, text='Reset to Default', command=self.reset_defaults).pack(fill='x')

        # Start update thread
        self._running = True
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()

    def on_canvas_click(self, event):
        # Map click to last faces and their region polygons
        faces = face_state.get_last_faces()
        if not faces:
            return
        h = int(self.canvas['height'])
        w = int(self.canvas['width'])
        # Use first face for convenience
        face = faces[0]
        lm = None
        try:
            if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
                lm = [(float(x), float(y)) for x, y in face.landmark_2d_106]
        except Exception:
            lm = None
        if lm is None:
            return
        # Determine which region contains the click
        px = event.x
        py = event.y
        # Canvas may be scaled relative to frame; we assume 1:1 for now
        for region in REGION_KEYS:
            indices = afc.REGION_GROUPS.get(region, [])
            poly = afc.polygon_from_indices(lm, indices)
            if poly.size == 0:
                continue
            # point-in-polygon
            if cv2.pointPolygonTest(poly.astype('int32'), (px, py), False) >= 0:
                # Increase slider slightly to indicate selection
                cur = get_current()
                cur_val = cur.get(region, DEFAULTS.get(region, 50))
                cur[region] = min(100, cur_val + 5)
                write_config(cur)
                return

    def apply_preset(self, name: str):
        preset = PRESETS.get(name)
        if not preset:
            return
        cur = get_current()
        # Merge preset values
        for k, v in preset.items():
            if k == 'advanced_controls':
                for kk, vv in v.items():
                    cur[kk] = vv
            else:
                cur[k] = v
        write_config(cur)

    def save_profile(self):
        name = simpledialog.askstring('Save Profile', 'Profile name:')
        if not name:
            return
        data = get_current()
        path = profile_manager.save_profile(name, data)
        messagebox.showinfo('Saved', f'Profile saved to {path}')

    def load_profile(self):
        profiles = profile_manager.list_profiles()
        if not profiles:
            messagebox.showinfo('No profiles', 'No profiles found')
            return
        name = simpledialog.askstring('Load Profile', f'Enter profile name (available: {", ".join(profiles)})')
        if not name:
            return
        data = profile_manager.load_profile(name)
        if not data:
            messagebox.showerror('Not found', 'Profile not found')
            return
        write_config(data)

    def reset_defaults(self):
        write_config(DEFAULTS)

    def _draw_preview(self):
        # Draw overlay based on last face and current config
        faces = face_state.get_last_faces()
        self.canvas.delete('all')
        if not faces:
            return
        face = faces[0]
        lm = None
        try:
            if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
                lm = [(float(x), float(y)) for x, y in face.landmark_2d_106]
        except Exception:
            lm = None
        if lm is None:
            return
        cfg = get_current()
        h = int(self.canvas['height'])
        w = int(self.canvas['width'])
        # Build masks per region and draw semi-transparent polygon for each region with intensity
        for region in REGION_KEYS:
            indices = afc.REGION_GROUPS.get(region, [])
            poly = afc.polygon_from_indices(lm, indices)
            if poly.size == 0:
                continue
            # Scale polygon to canvas if necessary (assume landmarks are in canvas coords)
            pts = [(int(x), int(y)) for x, y in poly.tolist()]
            # Draw filled polygon with alpha based on current slider
            strength = cfg.get(region, DEFAULTS.get(region, 50)) / 100.0
            color = '#%02x%02x%02x' % (int(255 * (1.0 - strength)), int(255 * strength), 64)
            try:
                self.canvas.create_polygon(*[coord for pt in pts for coord in pt], fill=color, outline='')
            except Exception:
                pass

    def _update_loop(self):
        while self._running:
            try:
                self._draw_preview()
            except Exception:
                pass
            time.sleep(POLL_INTERVAL)

    def destroy(self):
        self._running = False
        self.top.destroy()


# Auto attach when module imported and tkinter root exists
def start_embed_thread(timeout: float = 10.0):
    def waiter():
        import tkinter as tk
        start = time.time()
        while time.time() - start < timeout:
            try:
                root = tk._default_root
                if root is not None:
                    # attach panel
                    root.after(200, lambda: AdvancedPanel(root))
                    return
            except Exception:
                pass
            time.sleep(0.2)
    t = threading.Thread(target=waiter, daemon=True)
    t.start()


try:
    start_embed_thread()
except Exception:
    pass
