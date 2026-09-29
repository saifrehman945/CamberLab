"""Wall physics of scripts/airfrans/surface.py on analytic cases (no dataset needed)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.airfrans.geometry import cosine_grid, naca_coordinates  # noqa: E402
from scripts.airfrans.surface import (  # noqa: E402
    WallData,
    curves,
    integrate_forces,
    loop_normals,
    wall_frame,
    wall_gradients,
    wall_shear,
)


def _strip_mesh(nx: int = 21, ny: int = 6):
    """Structured triangles on [0, 1] x [0, 0.1]; the wall is the bottom row y = 0."""
    xs, ys = np.linspace(0, 1, nx), np.linspace(0, 0.1, ny)
    X, Y = np.meshgrid(xs, ys)
    x, y = X.ravel(), Y.ravel()
    node = lambda i, j: j * nx + i  # noqa: E731
    tris = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a, b, c, d = node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)
            tris += [[a, b, c], [a, c, d]]
    return x, y, np.asarray(tris), np.arange(nx)


def test_linear_shear_gradient_and_tau_are_exact():
    x, y, tris, wall = _strip_mesh()
    a, nu = 250.0, 1.5e-5
    sample = SimpleNamespace(x=x, y=y, triangles=tris, fields={"Ux": a * y, "Uy": np.zeros_like(y)})
    grad = wall_gradients(sample, wall)
    np.testing.assert_allclose(grad[:, 0, 1], a)
    np.testing.assert_allclose(grad[:, [0, 1, 1], [0, 0, 1]], 0.0, atol=1e-9)
    tau = wall_shear(grad, np.tile([0.0, 1.0], (len(wall), 1)), nu=nu)
    np.testing.assert_allclose(tau, np.tile([nu * a, 0.0], (len(wall), 1)), atol=1e-15)


def test_uniform_pressure_gives_no_force_and_uniform_shear_gives_drag():
    xy = naca_coordinates("2412")
    zero = np.zeros((len(xy), 2))
    f = integrate_forces(xy, np.full(len(xy), 7.3), zero, alpha_rad=0.1, u_inf=30.0)
    assert abs(f["Cl_int_p"]) < 1e-12 and abs(f["Cd_int_p"]) < 1e-12

    perimeter = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1).sum()
    u = 30.0
    f = integrate_forces(xy, np.zeros(len(xy)), np.tile([1.0, 0.0], (len(xy), 1)), 0.0, u)
    assert f["Cd_int_tau"] == pytest.approx(2 * perimeter / u**2)
    assert f["Cl_int_tau"] == pytest.approx(0.0, abs=1e-12)


def test_normals_point_into_the_fluid_for_either_loop_orientation():
    xy = naca_coordinates("0012")
    for loop in (xy, xy[::-1]):
        n = loop_normals(loop)
        top = np.argmax(loop[:, 1])
        assert n[top, 1] > 0.99
        nose = np.argmin(loop[:, 0])
        assert n[nose, 0] < -0.99


@pytest.mark.parametrize("code", ["0012", "2412", "23012"])
def test_frame_and_curves_on_analytic_sections(code):
    xy = naca_coordinates(code)
    x_c, upper, tangent = wall_frame(xy)
    assert x_c.min() >= -1e-9 and x_c.max() == pytest.approx(1.0)
    assert (xy[upper, 1].mean() > xy[~upper, 1].mean())
    # LE -> TE tangents point downstream away from the nose
    aft = x_c > 0.1
    assert (tangent[aft, 0] > 0).all()

    # Cp = x/c on both surfaces; attached shear along +t except reversed flow aft on the upper surface
    q = 0.5 * 40.0**2
    tau = tangent * 1e-3 * q
    reversed_ = upper & (x_c > 0.7)
    tau[reversed_] *= -1
    wall = WallData(xy=xy, p=x_c * q, tau=tau, tangent=tangent, x_c=x_c, upper=upper, q_inf=q)
    c = curves(wall)
    grid = cosine_grid()
    for side in ("upper", "lower"):
        np.testing.assert_allclose(c[f"Cp_{side}"], grid, atol=2e-3)
    inner = (grid > 0.05) & (grid < 0.95)
    assert (c["Cf_lower"][inner] > 0).all()
    assert (c["Cf_upper"][inner & (grid < 0.65)] > 0).all()
    assert (c["Cf_upper"][inner & (grid > 0.75)] < 0).all()
