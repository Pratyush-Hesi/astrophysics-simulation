"""
Astrophysics Simulation Suite
==============================
Three self-contained physics simulations:
  1. Kepler Orbital Mechanics   – numerical integration of planetary orbits
  2. N-Body Gravitational System – 3 stars pulling on each other
  3. Projectile on Alien Worlds  – compare surface gravity across planets/moons
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
import matplotlib.cm as cm

plt.rcParams.update({
    "figure.facecolor": "#0d0d1a",
    "axes.facecolor":   "#0d0d1a",
    "text.color":       "white",
    "axes.labelcolor":  "white",
    "xtick.color":      "#aaaacc",
    "ytick.color":      "#aaaacc",
    "axes.edgecolor":   "#333355",
    "grid.color":       "#1a1a2e",
    "grid.alpha":       0.6,
})

# ──────────────────────────────────────────────────────────────────
# SIMULATION 1 — KEPLER ORBITAL MECHANICS
# Uses Velocity Verlet integration to solve Newton's law of gravity
# for a star–planet system.  Plots the orbit and speed colour-map,
# demonstrating Kepler's 2nd law (faster near perihelion).
# ──────────────────────────────────────────────────────────────────

def sim_kepler(ax):
    """
    Simulate an elliptical orbit using Velocity Verlet integration.

    Physics:
        F = G·M·m / r²   (Newton's Law of Gravitation)
        a = F/m  →  a = -G·M / r³ · r⃗

    We work in 'simulation units':
        G·M = 1,  initial position = (1, 0),  initial velocity = (0, 0.7)
    This gives an ellipse with eccentricity ~0.5.
    """
    GM   = 1.0
    dt   = 1e-3
    steps = 30_000

    # State: position and velocity
    pos = np.array([1.0, 0.0])
    vel = np.array([0.0, 0.70])

    xs, ys, speeds = [], [], []

    for _ in range(steps):
        r     = np.linalg.norm(pos)
        acc   = -GM / r**3 * pos          # gravitational acceleration vector

        # Velocity Verlet step
        vel_half = vel + 0.5 * acc * dt
        pos      = pos + vel_half * dt
        r_new    = np.linalg.norm(pos)
        acc_new  = -GM / r_new**3 * pos
        vel      = vel_half + 0.5 * acc_new * dt

        xs.append(pos[0]);  ys.append(pos[1])
        speeds.append(np.linalg.norm(vel))

    xs, ys, speeds = np.array(xs), np.array(ys), np.array(speeds)

    # Colour the orbit by speed (Kepler's 2nd law: fast near star)
    points  = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segs    = np.concatenate([points[:-1], points[1:]], axis=1)
    norm    = plt.Normalize(speeds.min(), speeds.max())
    lc      = LineCollection(segs, cmap="plasma", norm=norm, linewidth=1.8, alpha=0.9)
    lc.set_array(speeds[:-1])
    ax.add_collection(lc)

    # Star at focus
    ax.scatter([0], [0], s=180, color="#FFD700", zorder=5, label="Star (focus)")

    # Mark perihelion & aphelion
    idx_peri = np.argmin(np.hypot(xs, ys))
    idx_aphe = np.argmax(np.hypot(xs, ys))
    ax.scatter(xs[idx_peri], ys[idx_peri], s=60, color="cyan",  zorder=6, label="Perihelion (fastest)")
    ax.scatter(xs[idx_aphe], ys[idx_aphe], s=60, color="orange",zorder=6, label="Aphelion (slowest)")

    cbar = plt.colorbar(lc, ax=ax, pad=0.02)
    cbar.set_label("Orbital Speed", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_aspect("equal")
    ax.set_xlim(-2.2, 1.5);  ax.set_ylim(-1.5, 1.5)
    ax.set_title("① Kepler Orbit — Velocity Verlet Integration\n"
                 "Colour = speed (Kepler's 2nd Law)", color="white", fontsize=11)
    ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="none", labelcolor="white")
    ax.grid(True)


# ──────────────────────────────────────────────────────────────────
# SIMULATION 2 — N-BODY GRAVITATIONAL SYSTEM (3 Stars)
# Uses the 4th-order Runge-Kutta method (RK4) to integrate three
# mutually interacting bodies.  The three-body problem is chaotic —
# small changes in initial conditions lead to wildly different orbits.
# ──────────────────────────────────────────────────────────────────

def _nbody_deriv(state, masses, G=1.0):
    """Return time-derivatives [v1,v2,v3, a1,a2,a3] for 3 bodies in 2D."""
    n = len(masses)
    pos = state[:n*2].reshape(n, 2)
    vel = state[n*2:].reshape(n, 2)
    acc = np.zeros_like(pos)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r_vec = pos[j] - pos[i]
            r     = np.linalg.norm(r_vec) + 1e-6   # softening
            acc[i] += G * masses[j] / r**3 * r_vec
    return np.concatenate([vel.flatten(), acc.flatten()])


def rk4_step(state, masses, dt):
    k1 = _nbody_deriv(state,          masses)
    k2 = _nbody_deriv(state + dt/2*k1, masses)
    k3 = _nbody_deriv(state + dt/2*k2, masses)
    k4 = _nbody_deriv(state + dt*k3,   masses)
    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


def sim_nbody(ax):
    """
    Simulate a classic figure-8 three-body orbit (Chenciner & Montgomery 2000).
    Three equal-mass stars chase each other along a shared figure-8 path.
    """
    # Figure-8 initial conditions (normalised units)
    m  = np.array([1.0, 1.0, 1.0])
    x0 = np.array([
        [ 0.97000436, -0.24308753],
        [-0.97000436,  0.24308753],
        [ 0.0,          0.0      ],
    ])
    v_3 = np.array([-0.93240737/2, -0.86473146/2])   # velocity of body 3 (conjugate momentum trick)
    v0  = np.array([
        [ 0.93240737/2,  0.86473146/2],
        [ 0.93240737/2,  0.86473146/2],
        [-0.93240737,   -0.86473146  ],
    ])

    state = np.concatenate([x0.flatten(), v0.flatten()])
    dt    = 5e-4
    steps = 60_000

    trails = [[] for _ in range(3)]
    colors = ["#FF6B6B", "#6BFFD0", "#FFD93D"]

    for step in range(steps):
        pos = state[:6].reshape(3, 2)
        for i in range(3):
            trails[i].append(pos[i].copy())
        state = rk4_step(state, m, dt)

    for i, (trail, col) in enumerate(zip(trails, colors)):
        t   = np.array(trail)
        # Fade colour along trail
        n   = len(t)
        fade = np.linspace(0.1, 1.0, n)
        pts  = t.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)

        # Build Nx4 RGBA array — vary alpha per segment for true fade
        base_rgba = to_rgba(col)
        rgba_colors = np.zeros((len(segs), 4))
        rgba_colors[:, :3] = base_rgba[:3]      # RGB constant
        rgba_colors[:, 3]  = fade[:-1] * 0.85   # alpha varies

        lc = LineCollection(segs, colors=rgba_colors, linewidth=0.8)
        ax.add_collection(lc)
        # Current position
        ax.scatter(*t[-1], s=80, color=col, zorder=5, label=f"Star {i+1}")

    ax.set_xlim(-1.5, 1.5);  ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal")
    ax.set_title("② Figure-8 Three-Body Orbit — RK4 Integration\n"
                 "Chaotic mutual gravity (Chenciner & Montgomery 2000)", color="white", fontsize=11)
    ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="none", labelcolor="white")
    ax.grid(True)


# ──────────────────────────────────────────────────────────────────
# SIMULATION 3 — PROJECTILE ON DIFFERENT WORLDS
# Solves standard projectile equations  y(t) = v₀·sin(θ)·t − ½g·t²
# for several solar-system bodies, showing how surface gravity
# (g) changes the range and height of a thrown ball.
# ──────────────────────────────────────────────────────────────────

def sim_projectile(ax):
    """
    Compare projectile trajectories on seven worlds.

    Equations:
        x(t) = v₀ · cos(θ) · t
        y(t) = v₀ · sin(θ) · t  −  ½ · g · t²
        Range = v₀² · sin(2θ) / g
    """
    bodies = {
        "Moon":    1.62,
        "Mars":    3.72,
        "Earth":   9.81,
        "Titan":   1.35,
        "Jupiter": 24.79,
        "Venus":   8.87,
        "Pluto":   0.62,
    }
    v0    = 30.0       # m/s  initial speed
    theta = np.radians(45.0)   # optimal angle

    palette = plt.get_cmap("cool", len(bodies))

    for idx, (name, g) in enumerate(sorted(bodies.items(), key=lambda x: x[1])):
        t_flight = 2 * v0 * np.sin(theta) / g
        t        = np.linspace(0, t_flight, 500)
        x        = v0 * np.cos(theta) * t
        y        = v0 * np.sin(theta) * t - 0.5 * g * t**2
        color    = palette(idx / (len(bodies) - 1))
        ax.plot(x, y, color=color, linewidth=2.0, label=f"{name}  (g={g} m/s²)")

    ax.axhline(0, color="#555577", linewidth=0.8)
    ax.set_xlabel("Horizontal Distance (m)", color="white")
    ax.set_ylabel("Height (m)", color="white")
    ax.set_title("③ Projectile Motion Across the Solar System\n"
                 f"v₀ = {v0} m/s, θ = 45° — same throw, different worlds",
                 color="white", fontsize=11)
    ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="none",
              labelcolor="white", loc="upper right")
    ax.grid(True)


# ──────────────────────────────────────────────────────────────────
# COMPOSE ALL THREE PLOTS
# ──────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 13))
fig.patch.set_facecolor("#0d0d1a")

gs  = gridspec.GridSpec(2, 2, figure=fig,
                        hspace=0.38, wspace=0.30,
                        left=0.07, right=0.97,
                        top=0.93, bottom=0.07)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])   # full-width bottom panel

sim_kepler(ax1)
sim_nbody(ax2)
sim_projectile(ax3)

fig.suptitle("Astrophysics Simulation Suite  |  NumPy + Matplotlib",
             color="white", fontsize=15, fontweight="bold", y=0.97)

plt.savefig("/mnt/user-data/outputs/astro_simulations.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("Saved → astro_simulations.png")
plt.show()
