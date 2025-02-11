
import math
from math import radians, cos, sin
import numpy as np

def Compute_angle(values):
    # Extract axes from the input dictionary
    x_axis = values['x_axis']
    y_axis = values['y_axis']
    z_axis = values['z_axis']

    # Rotation matrix components from the axes
    R11, R21, R31 = z_axis['x'], y_axis['x'], x_axis['x']
    R32, R33 = x_axis['z'], x_axis['y']

    # Compute the Euler angles for a ZYX rotation
    # a = atan2(R21, R11)
    a = math.atan2(R21, R11)

    # b = asin(-R31)
    b = math.asin(-R31)

    # c = atan2(R32, R33)
    c = math.atan2(R32, R33)

    # Convert angles from radians to degrees for better interpretation
    a_deg = math.degrees(a)
    b_deg = math.degrees(b)
    c_deg = math.degrees(c)

    return a_deg, b_deg, c_deg

def rotate_point(point, rotation):
    x, y, z = point
    rx, ry, rz = map(radians, rotation)  # Convert rotation angles from degrees to radians

    # Rotate around the X axis
    y, z = y * cos(rx) - z * sin(rx), y * sin(rx) + z * cos(rx)

    # Rotate around the Y axis
    x, z = x * cos(ry) + z * sin(ry), z * cos(ry) - x * sin(ry)

    # Rotate around the Z axis
    x, y = x * cos(rz) - y * sin(rz), x * sin(rz) + y * cos(rz)

    return (x, y, z)


def rotate_point_singleX(point, new_x_axis):
    # Normalize the new x-axis vector
    new_x_axis = np.array(new_x_axis)
    norm_x_axis = new_x_axis / np.linalg.norm(new_x_axis)

    # Standard x-axis
    standard_x_axis = np.array([1, 0, 0])

    # Compute the rotation axis (cross product of standard x-axis and new x-axis)
    rotation_axis = np.cross(standard_x_axis, norm_x_axis)

    # Compute the angle between the standard x-axis and new x-axis
    angle = np.arccos(np.dot(standard_x_axis, norm_x_axis))

    # Create the rotation matrix using the axis-angle formula (Rodrigues' rotation formula)
    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ])
    rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    # Convert point to a numpy array
    point_vector = np.array(point)
    
    # Perform matrix multiplication to rotate the point
    rotated_point = np.dot(rotation_matrix, point_vector)
    
    return rotated_point

def check_normal_direction(normal):
    for component in normal:
        if component > 0:
            return 1
        elif component < 0:
            return -1
    return 0


def translate_local(point, local_translation):
    point = (point[0] + local_translation[0], 
                point[1] + local_translation[1], 
                point[2] + local_translation[2])
    return point

def translate_global(point, global_translation):
    point = (point[0] + global_translation['x'], 
                point[1] + global_translation['y'], 
                point[2] + global_translation['z'])
    return point