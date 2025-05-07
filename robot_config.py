# robot_config.py

"""
Robot Configuration File

This file contains all the necessary parameters for the robot simulation,
including joint mappings, torque values, position and velocity gains,
and friction coefficients.
"""

import math

# ------------------------ Joint Mappings ------------------------ #
# Mapping of joint indices to leg joints
# 0: Upper Back Right Leg
# 1: Lower Back Right Leg
# 2: Upper Front Right Leg
# 3: Lower Front Right Leg
# 4: Upper Front Left Leg
# 5: Lower Front Left Leg
# 6: Upper Back Left Leg
# 7: Lower Back Left Leg

JOINT_MAPPING = {
    0: "Upper Back Right Leg",
    1: "Lower Back Right Leg",
    2: "Upper Front Right Leg",
    3: "Lower Front Right Leg",
    4: "Upper Front Left Leg",
    5: "Lower Front Left Leg",
    6: "Upper Back Left Leg",
    7: "Lower Back Left Leg"
}

# ------------------------ Torque Values ------------------------ #
# Motor Specifications
# Maximum Torque per Motor: 17 kg.cm
# Link Length: 13 mm = 0.013 m
# Torque (N.m) = Torque (kg.cm) * 0.0980665
MAX_TORQUE_KG_CM = 17
MAX_TORQUE_NM = MAX_TORQUE_KG_CM * 0.0980665  # ≈1.667 N.m

# Calculating Maximum Force per Joint
# Force (N) = Torque (N.m) / Link Length (m)
LINK_LENGTH = 0.013  # 13 mm in meters
MAX_FORCE = MAX_TORQUE_NM / LINK_LENGTH  # ≈128 N

# ------------------------ Motor Force Limits ------------------------ #
# Maximum force each motor can apply
HIP_MAX_FORCE = MAX_FORCE
KNEE_MAX_FORCE = MAX_FORCE

# ------------------------ Gains ------------------------ #
# Position and Velocity Gains
# Suggested realistic values based on servo specifications
POSITION_GAIN = 0.8  # Between 0.5 and 1.0 for responsiveness without overshooting
VELOCITY_GAIN = 0.8  # Between 0.5 and 1.0 for smooth velocity control


# ------------------------ Additional Parameters ------------------------ #
# Gait Parameters Ranges (degrees)
# gait is a+b(omega t +c) for each motor
# we restrict motors so max angle away from nowm is always 30.
A_MIN, A_MAX = -15, 15             # Range for 'a' parameters
B_MIN, B_MAX = -15, 15             # Range for 'b' parameters
C_MIN, C_MAX = 0, 2 * math.pi    # Range for 'c' parameters
OMEGA_MIN, OMEGA_MAX = 0.5, 10   # Range for 'omega' parameter

# ------------------------ Friction Coefficients ------------------------ #
# Friction between robot feet and the floor
FLOOR_FRICTION_COEFFICIENT = 0.5  # Very low friction to mimic slipping

