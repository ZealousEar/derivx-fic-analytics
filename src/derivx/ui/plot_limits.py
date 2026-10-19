"""Zoom and pan limits for the app's Plotly charts.

Plotly has no layout option for a minimum zoom span or for 3D camera limits,
so charts are bounded in two layers:

1. Declarative, always on: 2D axes get ``minallowed``/``maxallowed`` so they
   cannot be zoomed out or panned past the data (no negative strikes) and the
   3D scene gets a fixed aspect ratio.
2. An in-page guard (``st.components.v2``) that clamps every change to the
   limits stored in ``layout.meta["zoom"]``: a minimum 2D span, and the 3D
   zoom scale, camera distance and pan offset. It also makes wheel zoom
   cooperative: a plain wheel scrolls the page, while pinch or Ctrl/Cmd +
   wheel zooms the chart. Without components v2 the app turns wheel zoom off
   and relies on layer 1 alone.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

try:  # Streamlit >= 1.51
    from streamlit.components import v2 as _components_v2
except ImportError:  # pragma: no cover - depends on the installed Streamlit
    _components_v2 = None

SCENE_ASPECT = {"x": 1.0, "y": 1.0, "z": 0.9}

_GUARD_JS = r"""
export default function () {
  const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
  const busy = new WeakSet();

  function axisUpdates(gd, axes) {
    const upd = {};
    for (const [name, lim] of Object.entries(axes || {})) {
      const ax = gd._fullLayout[name];
      // Linear and category axes have numeric ranges; date axes are bounded
      // declaratively by minallowed/maxallowed only.
      if (!ax || !["linear", "category"].includes(ax.type) || !Array.isArray(ax.range)) continue;
      const a = Number(ax.range[0]), b = Number(ax.range[1]);
      if (!isFinite(a) || !isFinite(b)) continue;
      const span = clamp(b - a, lim.min_span, lim.max - lim.min);
      let lo = (a + b) / 2 - span / 2, hi = lo + span;
      if (lo < lim.min) { hi += lim.min - lo; lo = lim.min; }
      if (hi > lim.max) { lo -= hi - lim.max; hi = lim.max; }
      const tol = 1e-9 * (lim.max - lim.min);
      if (Math.abs(lo - a) > tol || Math.abs(hi - b) > tol) upd[name + ".range"] = [lo, hi];
    }
    return upd;
  }

  function sceneUpdates(gd, s) {
    const L = gd._fullLayout.scene;
    if (!L || !L._scene) return {};
    const upd = {};
    // Orthographic zoom rescales the aspect ratio uniformly.
    const k = L.aspectratio.x / s.aspect.x, kc = clamp(k, s.zoom[0], s.zoom[1]);
    if (Math.abs(kc - k) > 1e-6) {
      upd["scene.aspectratio"] = { x: s.aspect.x * kc, y: s.aspect.y * kc, z: s.aspect.z * kc };
    }
    // Perspective / drag zoom moves the eye; pan moves the centre.
    const cam = L._scene.getCamera();
    const c = { ...cam.center }, e = { ...cam.eye };
    let moved = false;
    for (const a of ["x", "y", "z"]) {
      const v = clamp(c[a], -s.pan, s.pan);
      if (v !== c[a]) { e[a] += v - c[a]; c[a] = v; moved = true; }
    }
    const d = [e.x - c.x, e.y - c.y, e.z - c.z], r = Math.hypot(...d);
    const rc = clamp(r, s.distance[0], s.distance[1]);
    if (r > 0 && Math.abs(rc - r) > 1e-6) {
      e.x = c.x + d[0] * rc / r; e.y = c.y + d[1] * rc / r; e.z = c.z + d[2] * rc / r;
      moved = true;
    }
    if (moved) upd["scene.camera"] = { eye: e, center: c, up: cam.up, projection: cam.projection };
    return upd;
  }

  function enforce(gd) {
    const lim = gd.layout && gd.layout.meta && gd.layout.meta.zoom;
    if (!lim || busy.has(gd) || !window.Plotly || !gd._fullLayout) return;
    const upd = lim.scene ? sceneUpdates(gd, lim.scene) : axisUpdates(gd, lim.axes);
    if (!Object.keys(upd).length) return;
    busy.add(gd);
    Promise.resolve(window.Plotly.relayout(gd, upd))
      .catch(() => {})
      .finally(() => busy.delete(gd));
  }

  const zoomSoon = (gd) => {
    clearTimeout(gd.__derivxZoomTimer);
    gd.__derivxZoomTimer = setTimeout(() => enforce(gd), 150);
  };
  const charts = () => Array.from(document.querySelectorAll(".js-plotly-plot"))
    .filter((gd) => gd.layout && gd.layout.meta && gd.layout.meta.zoom);
  const scenes = () => charts().filter((gd) => gd.layout.meta.zoom.scene);

  function attach() {
    for (const gd of charts()) {
      if (!gd.__derivxZoomWheel) {
        // Cooperative wheel: a plain wheel never reaches Plotly, so the page
        // scrolls; pinch (reported as ctrl+wheel) or Ctrl/Cmd+wheel zooms.
        gd.__derivxZoomWheel = true;
        gd.addEventListener("wheel", (e) => {
          if (e.ctrlKey || e.metaKey) zoomSoon(gd);
          else e.stopPropagation();
        }, { capture: true, passive: true });
      }
      // Re-attach whenever Plotly replaced the element's event emitter.
      if (typeof gd.on !== "function" || gd.__derivxZoomEv === gd._ev) continue;
      gd.__derivxZoomEv = gd._ev;
      gd.on("plotly_relayout", () => enforce(gd));
      enforce(gd);
    }
  }

  // 3D middle/right-button drags emit no relayout, so re-check every scene
  // once the camera has settled after a release, and poll as a backstop
  // (never mid-drag, so the guard does not fight the gesture).
  let pressed = 0;
  const onDown = () => { pressed += 1; };
  const onUp = () => {
    pressed = 0;
    for (const delay of [80, 450]) setTimeout(() => scenes().forEach(enforce), delay);
  };
  window.addEventListener("pointerdown", onDown, true);
  window.addEventListener("pointerup", onUp, true);
  window.addEventListener("pointercancel", onUp, true);

  attach();
  const timer = setInterval(() => {
    attach();
    if (!pressed) scenes().forEach(enforce);
  }, 500);
  return () => {
    clearInterval(timer);
    window.removeEventListener("pointerdown", onDown, true);
    window.removeEventListener("pointerup", onUp, true);
    window.removeEventListener("pointercancel", onUp, true);
  };
}
"""

_GUARD = (
    _components_v2.component("derivx_zoom_guard", js=_GUARD_JS)
    if _components_v2 is not None
    else None
)
GUARD_AVAILABLE = _GUARD is not None


def axis_limits(
    values: Iterable[float],
    pad: float = 0.05,
    floor: float | None = None,
    min_span_frac: float = 0.02,
) -> Dict[str, float]:
    """Padded axis bounds around ``values`` plus the smallest allowed span.

    Non-finite values are ignored. ``floor`` clips the lower bound (e.g. 0 for
    strikes or vols); ``min_span_frac`` sets how far in the axis can zoom, as
    a fraction of the full bounded width.
    """

    arr = np.asarray(list(values), dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    lo, hi = (float(arr.min()), float(arr.max())) if arr.size else (0.0, 1.0)
    span = hi - lo
    if span <= 0:  # flat data: give the axis some room around the level
        span = max(abs(hi) * 0.1, 1e-6)
        lo, hi = lo - span / 2, hi + span / 2
    lo, hi = lo - span * pad, hi + span * pad
    if floor is not None and lo < floor:
        lo = floor
        hi = max(hi, lo + span)
    return {"min": lo, "max": hi, "min_span": (hi - lo) * min_span_frac}


def _merge_meta(fig: go.Figure, zoom: Mapping[str, object]) -> None:
    meta = fig.layout.meta if isinstance(fig.layout.meta, dict) else {}
    fig.update_layout(meta={**meta, "zoom": dict(zoom)})


def bound_axes(fig: go.Figure, **axes: Mapping[str, object]) -> go.Figure:
    """Start each named axis (``xaxis``, ``yaxis2``, ...) at its limits and
    forbid zooming or panning past them.

    Limits are ``{"min", "max", "min_span"}`` as from :func:`axis_limits`;
    date axes take ISO strings for ``min``/``max`` and no ``min_span``.
    """

    for name, lim in axes.items():
        fig.update_layout(
            {
                name: dict(
                    range=[lim["min"], lim["max"]],
                    minallowed=lim["min"],
                    maxallowed=lim["max"],
                    autorange=False,
                )
            }
        )
    _merge_meta(fig, {"axes": axes})
    return fig


def bound_scene(
    fig: go.Figure,
    eye: Mapping[str, float],
    zoom: Tuple[float, float] = (0.6, 2.5),
    pan: float = 0.35,
) -> go.Figure:
    """Fix the 3D aspect ratio and record zoom/pan limits for the guard.

    ``zoom`` is the (min, max) magnification relative to the default view;
    ``pan`` bounds each camera-centre coordinate in normalised scene units.
    """

    fig.update_scenes(aspectmode="manual", aspectratio=SCENE_ASPECT)
    r0 = math.sqrt(sum(float(v) ** 2 for v in eye.values()))
    _merge_meta(
        fig,
        {
            "scene": {
                "aspect": dict(SCENE_ASPECT),
                "zoom": [zoom[0], zoom[1]],
                "distance": [r0 / zoom[1], r0 / zoom[0]],
                "pan": pan,
            }
        },
    )
    return fig


def mount_zoom_guard() -> bool:
    """Mount the relayout guard once per script run; False if unavailable."""

    if _GUARD is None:
        return False
    with st.sidebar:
        _GUARD(key="derivx_zoom_guard")
    return True
