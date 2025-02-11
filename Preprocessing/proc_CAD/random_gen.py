import numpy as np
import json

def generate_random_rectangle(perpendicular_axis):
    # Generate a random center point for the rectangle
    center = np.random.uniform(-10, 10, 3)
    
    # Generate random length and width
    length = np.random.uniform(5, 10)
    width = np.random.uniform(5, 10)
    
    # Calculate the half lengths
    half_length = length / 2
    half_width = width / 2

    # Initialize points array
    points = np.zeros((4, 3))

    # Randomly choose a direction for the normal vector (+1 or -1)
    normal_direction = np.random.choice([-1, 1])

    if perpendicular_axis == 'x':
        # Points in the YZ plane, X is constant
        constant_value = center[0]
        normal_vector = [normal_direction, 0, 0]  # Normal vector along x-axis
        points[:, 0] = constant_value
        points[:, 1] = [center[1] - half_length, center[1] - half_length, center[1] + half_length, center[1] + half_length]
        points[:, 2] = [center[2] - half_width, center[2] + half_width, center[2] + half_width, center[2] - half_width]
    elif perpendicular_axis == 'y':
        # Points in the XZ plane, Y is constant
        constant_value = center[1]
        normal_vector = [0, normal_direction, 0]  # Normal vector along y-axis
        points[:, 1] = constant_value
        points[:, 0] = [center[0] - half_length, center[0] - half_length, center[0] + half_length, center[0] + half_length]
        points[:, 2] = [center[2] - half_width, center[2] + half_width, center[2] + half_width, center[2] - half_width]
    elif perpendicular_axis == 'z':
        # Points in the XY plane, Z is constant
        constant_value = center[2]
        normal_vector = [0, 0, normal_direction]  # Normal vector along z-axis
        points[:, 2] = constant_value
        points[:, 0] = [center[0] - half_length, center[0] - half_length, center[0] + half_length, center[0] + half_length]
        points[:, 1] = [center[1] - half_width, center[1] + half_width, center[1] + half_width, center[1] - half_width]
    else:
        raise ValueError("Invalid axis. Choose 'x', 'y', or 'z'.")

    return points, normal_vector


def generate_random_extrude():
    if np.random.choice([True, False]):
        random_float = np.random.uniform(5, 10)
    else:
        random_float = np.random.uniform(-4, -2)
    return random_float


def generate_random_fillet():
    random_float = np.random.uniform(1, 2)
    return random_float
 
def generate_random_cylinder_radius():
    random_float = np.random.uniform(3, 5)
    return random_float
