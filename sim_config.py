# sim_config.py
FPS = 60

DOMAIN_SIZE_M = 50_000
AREA_SIZE_KM = 50.0
SPAWN_BUFFER_M = 1_000
MAX_PARCELS = 4
SPAWN_PROBABILITY = 0.10  # Increased from 0.04 for quicker refills
SINGLE_CLOUD_MODE = False
OPACITY_RAMP = 0.01
MAX_OPACITY = 0.95
CLOUD_WRAP_AROUND = True
FORCE_INITIAL_CLOUD = True
POSITION_HISTORY_LENGTH = 8
CLOUD_TRANSMITTANCE = 0.2
SHADOW_FADE_MS = 500
PENUMBRA_WIDTH = 60
MIN_CLOUD_ALTITUDE = 0.2
MAX_CLOUD_ALTITUDE = 5.0
ALT_PARALLAX_K = 0.2
WIND_UPDATE_SEC = 5
PHYSICS_TIMESTEP = 1.0/60.0
WIND_SMOOTH = 0.95  # Increased from 0.8 for more velocity inertia

# Wind direction and speed
BASE_WIND_SPEED = 2.0  # Reduced from 3.0 for slower movement
BASE_WIND_DIRECTION = 135.0  # North-West to South-East

DEFAULT_WIND_SPEED = 2.0  # Reduced from 3.0
DEFAULT_WIND_DIRECTION = 135.0  # North-West to South-East (degrees)

# --- default fallback wind speed (used in cloud movement if nothing else provided)
DEFAULT_WIND_SPEED_KMH = 2.0  # Reduced from 3.0
DEFAULT_WIND_DIRECTION_DEG = 135.0

WIND_GRID = 256
MOVEMENT_MULTIPLIER = 5.0  # Reduced from 10.0 for slower movement
INTEGRATION_DEGREE = 4
LAYER_HEIGHTS = [0.0, 0.25, 1.0, 8.0]
LAYER_SPEED_FACTORS = [1.2, 1.0, 0.8, 0.6]
# Override direction offsets to zero for all layers
LAYER_DIRECTION_OFFSETS = [0.0, 0.0, 0.0, 0.0]
PANEL_SIZE_KM = 0.4

# Cloud lifecycle parameters - INCREASED VALUES
CLOUD_GROWTH_FRAMES = 3600    # Doubled from 1800 (60 seconds at 60 FPS)
CLOUD_STABLE_FRAMES = 7200    # Doubled from 3600 (120 seconds at 60 FPS)
CLOUD_DECAY_FRAMES = 3600     # Doubled from 1800 (60 seconds at 60 FPS)
CLOUD_BREATHING = 0.05
CLOUD_RANDOM_REMOVAL_CHANCE = 0.00001  # Reduced from 0.0001

# Cloud scattering parameters
SCATTER_PROBABILITY = 0.1
SCATTER_FRAGMENTS = (2, 4)

# Wind turbulence control for smoother movement
WIND_TURBULENCE = 0.05  # Reduced from 0.15

# Cloud type definitions
CLOUD_TYPES = {
    "cirrus": {
        "alt_km": 8.0,
        "r_km": (1.5, 2.5),
        "opacity_max": 0.30,
        "speed_k": 0.5
    },
    "cumulus": {
        "alt_km": 1.5,
        "r_km": (0.4, 1.0),
        "opacity_max": 0.70,
        "speed_k": 1.0
    },
    "cumulonimbus": {
        "alt_km": 2.0,
        "r_km": (3.0, 5.0),
        "opacity_max": 0.90,
        "speed_k": 0.8
    },
}

# Maximum cloud radius
R_MAX_KM = 6.0

# Weight for random cloud type selection
CLOUD_TYPE_WEIGHTS = {"cirrus": 1, "cumulus": 6, "cumulonimbus": 1}

# Puff count ranges for different cloud types
PUFF_MIN = {"cirrus": 3, "cumulus": 5, "cumulonimbus": 10}
PUFF_MAX = {"cirrus": 4, "cumulus": 8, "cumulonimbus": 14}

# --- Cloud Type Appearance Profiles ------------------------
# Defines the visual characteristics of different cloud types
CLOUD_TYPE_PROFILES = {
    "cirrus": {
        "cw": (2000, 4000),  # Long, thin
        "ch": (300, 800),
        "opacity": (0.2, 0.4),
        "rotation": (-0.3, 0.3)
    },
    "cumulus": {
        "cw": (1500, 2500),  # Round, fluffy
        "ch": (1500, 2500),
        "opacity": (0.5, 0.8),
        "rotation": (-0.1, 0.1)
    },
    "cumulonimbus": {
        "cw": (2000, 3000),  # Towering
        "ch": (3000, 5000),
        "opacity": (0.7, 1.0),
        "rotation": (-0.05, 0.05)
    }
}