Astrophysics Simulation Suite
Three self-contained physics simulations built with Python, NumPy, and Matplotlib — exploring orbital mechanics, gravitational dynamics, and projectile motion across the solar system.

Simulations:

Kepler Orbital Mechanics
Numerically integrates Newton's law of gravitation using the Velocity Verlet algorithm — a symplectic method that conserves energy over long runs. The resulting elliptical orbit is coloured by orbital speed, making Kepler's 2nd Law (equal areas in equal times) directly visible: the planet blazes through perihelion and crawls at aphelion.

Figure-8 Three-Body Orbit
Integrates three mutually attracting equal-mass stars using 4th-order Runge-Kutta (RK4). Uses the famous Chenciner & Montgomery (2000) initial conditions — one of the only known stable, periodic solutions to the otherwise chaotic three-body problem. Each star's trail fades in opacity over time using per-segment RGBA colour arrays.

Projectile Motion Across the Solar System
Solves kinematic equations analytically for seven solar system bodies (Earth, Moon, Mars, Venus, Jupiter, Titan, Pluto) with identical launch conditions. Demonstrates starkly how surface gravity governs range and hang time — the same throw that travels ~90 m on Earth covers over 2 km on Pluto.

Stack:
numpy — numerical integration, vector math
matplotlib — rendering, LineCollection for gradient trails, colormaps

Run:
bash
pip install numpy matplotlib
python astro_simulations.py
Output is saved as astro_simulations.png alongside a live plot window.

Integrators used: Velocity Verlet (Kepler), RK4 (N-body). All simulations run in dimensionless or SI units with inline documentation of the governing equations.

<img width="2559" height="1847" alt="astro_simulations (1)" src="https://github.com/user-attachments/assets/f4d1b2b0-221c-4010-ab53-9d14938c5c55" />

