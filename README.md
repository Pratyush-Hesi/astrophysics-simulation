Astrophysics Simulation Suite
Three self-contained physics simulations built with Python, NumPy, and Matplotlib — exploring orbital mechanics, gravitational dynamics, and projectile motion across the solar system.
Simulations
① Kepler Orbital Mechanics
Numerically integrates Newton's law of gravitation using the Velocity Verlet algorithm — a symplectic method that conserves energy over long runs. The resulting elliptical orbit is coloured by orbital speed, making Kepler's 2nd Law (equal areas in equal times) directly visible: the planet blazes through perihelion and crawls at aphelion.
② Figure-8 Three-Body Orbit
Integrates three mutually attracting equal-mass stars using 4th-order Runge-Kutta (RK4). Uses the famous Chenciner & Montgomery (2000) initial conditions — one of the only known stable, periodic solutions to the otherwise chaotic three-body problem. Each star's trail fades in opacity over time using per-segment RGBA colour arrays.
③ Projectile Motion Across the Solar System
Solves kinematic equations analytically for seven solar system bodies (Earth, Moon, Mars, Venus, Jupiter, Titan, Pluto) with identical launch conditions. Demonstrates starkly how surface gravity governs range and hang time — the same throw that travels ~90 m on Earth covers over 2 km on Pluto.
Stack

numpy — numerical integration, vector math
matplotlib — rendering, LineCollection for gradient trails, colormaps
