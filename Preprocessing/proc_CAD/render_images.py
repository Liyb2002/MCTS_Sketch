import Preprocessing.proc_CAD.brep_read
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import Preprocessing.proc_CAD.helper
import random
import numpy as np

import glob
import re

def find_bounding_box(edges_features):
    min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
    max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')

    for edge_info in edges_features:
        points = edge_info['vertices']
        if edge_info['is_curve'] and edge_info['sampled_points']:
            points += edge_info['sampled_points']

        for x, y, z in points:
            if x < min_x: min_x = x
            if y < min_y: min_y = y
            if z < min_z: min_z = z
            if x > max_x: max_x = x
            if y > max_y: max_y = y
            if z > max_z: max_z = z

    bounding_box_vertices = [
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z]
    ]

    bounding_box_edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),  # Bottom face edges
        (4, 5), (5, 7), (7, 6), (6, 4),  # Top face edges
        (0, 4), (1, 5), (2, 6), (3, 7)   # Side edges connecting bottom and top faces
        ]

    for edge in bounding_box_edges:
        edge_feature = {
            'vertices': [bounding_box_vertices[edge[0]], bounding_box_vertices[edge[1]]],
            'type': 'scaffold',
            'is_curve': False,
            'sampled_points': [],
            'projected_edge': [],
            'sigma' : 0.0,
            'mu': 0.0
        }
        edges_features.append(edge_feature)


    object_center = [(min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2]

    add_grid_lines(edges_features, min_x, min_y, min_z, max_x, max_y, max_z)

    return edges_features, object_center


def add_grid_lines(edges_features, min_x, min_y, min_z, max_x, max_y, max_z):
    # Define a distance threshold to determine "far from" the bounding box
    distance_threshold = 0.2  # Define the threshold as needed

    # Collect all vertices from edges_features
    all_vertices = []
    for edge in edges_features:
        all_vertices.extend(edge['vertices'])

    # Check each vertex if it is outside the bounding box by the distance threshold
    for vertex in all_vertices:
        x, y, z = vertex
        # Check distance from bounding box and add grid lines if necessary
        if x < min_x - distance_threshold or x > max_x + distance_threshold:
            new_edge = {
                'vertices': [(x, min_y, z), (x, max_y, z)],  # Vertical grid line
                'type': 'grid_line',
                'is_curve': False,
                'sampled_points': [],
                'projected_edge': [(x, min_y, z), (x, max_y, z)],
                'sigma': 0.0,
                'mu': 0.0
            }
            print("aaa")
            edges_features.append(new_edge)

        if y < min_y - distance_threshold or y > max_y + distance_threshold:
            new_edge = {
                'vertices': [(min_x, y, z), (max_x, y, z)],  # Horizontal grid line
                'type': 'grid_line',
                'is_curve': False,
                'sampled_points': [],
                'projected_edge': [(min_x, y, z), (max_x, y, z)],
                'sigma': 0.0,
                'mu': 0.0
            }
            print("aaa")
            edges_features.append(new_edge)

        if z < min_z - distance_threshold or z > max_z + distance_threshold:
            new_edge = {
                'vertices': [(x, y, min_z), (x, y, max_z)],  # Depth grid line
                'type': 'grid_line',
                'is_curve': False,
                'sampled_points': [],
                'projected_edge': [(x, y, min_z), (x, y, max_z)],
                'sigma': 0.0,
                'mu': 0.0
            }
            print("aaa")
            edges_features.append(new_edge)

def plot(edges_features):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for edge_info in edges_features:
        if not edge_info['is_curve']:
            xs, ys, zs = zip(*edge_info['vertices'])
            if edge_info['type'] == 'scaffold':
                ax.plot(xs, ys, zs, marker='o', color='green', label='Construction Line')
            else:
                ax.plot(xs, ys, zs, marker='o', color='red', label='Vertices')  

        # Plot curved edges with sampled points if they exist
        if edge_info['is_curve'] and edge_info['sampled_points']:
            xp, yp, zp = zip(*edge_info['sampled_points'])
            ax.plot(xp, yp, zp, linestyle='--', color='blue', label='Sampled Points')


    # Setting labels
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')

    # Show plot
    plt.show()

def optimize_opacities(edges_features, stylesheet):
    for edge_info in edges_features:
        edge_type = edge_info['type']

        if edge_type == 'scaffold':
            edge_info['mu'] = stylesheet["opacities_per_type"]["scaffold"]["mu"]
            edge_info['sigma'] = stylesheet["opacities_per_type"]["scaffold"]["sigma"]
            
        if edge_type == 'feature_line':
            edge_info['mu'] = stylesheet["opacities_per_type"]["vis_edges"]["mu"]
            edge_info['sigma'] = stylesheet["opacities_per_type"]["vis_edges"]["sigma"]
        
        if edge_type == 'grid_line':
            edge_info['mu'] = stylesheet["opacities_per_type"]["silhouette"]["mu"]
            edge_info['sigma'] = stylesheet["opacities_per_type"]["silhouette"]["sigma"]

        opacity = np.random.normal(loc=edge_info['mu'], scale=edge_info['sigma']/2, size=1)[0]
        opacity = max(0.0, min(1.0, opacity))
        edge_info['opacity'] = opacity


    return edges_features

def project_points(edges_features, obj_center):
    edges_features = Preprocessing.proc_CAD.helper.project_points(edges_features, obj_center)
    return edges_features

def overshoot_stroke(edges_features, factor = 10):
    for edge_info in edges_features:
        if edge_info['is_curve']:
            continue
        
        projected_edge = edge_info['projected_edge']
        point_1 = projected_edge[0]
        point_2 = projected_edge[1]

        # Calculate the direction vector from point_1 to point_2
        direction = [point_2[i] - point_1[i] for i in range(len(point_1))]
        
        # Normalize the direction vector
        length = sum(d**2 for d in direction)**0.5
        direction = [d / length for d in direction]
        
        # Generate a small random overshoot factor
        overshoot_factor = factor * random.random()  # Adjust the factor scale as needed

        point_1 = [point_1[i] - direction[i] * overshoot_factor for i in range(len(point_1))]
        point_2 = [point_2[i] + direction[i] * overshoot_factor for i in range(len(point_2))]

        edge_info['projected_edge'] = [point_1, point_2]


    return edges_features

def plot_2d(edges_features, directory):
    fig = plt.figure(figsize=(8, 6))

    for edge_info in edges_features:
        f_line = edge_info['projected_edge']
        opacity = edge_info['opacity']

        if edge_info['is_curve']:
            x_values = [point[0] for point in f_line]
            y_values = [point[1] for point in f_line]
        else:
            point1 = f_line[0]
            point2 = f_line[1]
            x_values = [point1[0], point2[0]]
            y_values = [point1[1], point2[1]]

        plt.plot(x_values, y_values, c="black", alpha=opacity * 0.5)

    filepath = os.path.join(directory, 'rendered_image.png')
    plt.savefig(filepath)
    plt.close(fig)


def perturb_strokes(edges_features, noise_level=1.0):
    for edge_info in edges_features:
        if edge_info['is_curve']:
            continue  # Assuming we only perturb straight line segments here for simplicity
        projected_edge = edge_info['projected_edge']
        perturbed_edge = []
        for point in projected_edge:
            # Generate random noise
            noise = np.random.normal(0, noise_level, 2)
            # Perturb the original point by adding noise
            perturbed_point = (point[0] + noise[0], point[1] + noise[1])
            perturbed_edge.append(perturbed_point)
        # Update the projected_edge with the perturbed points
        edge_info['projected_edge'] = perturbed_edge
    

def get_last_file(directory):
    
    # Construct the search pattern
    search_pattern = os.path.join(directory, 'canvas', 'brep_*.step')
    
    # Get the list of files matching the pattern
    step_files = glob.glob(search_pattern)

    step_indices = [int(re.search(r'brep_(\d+).step', file).group(1)) for file in step_files]
    largest_index = max(step_indices)

    largest_step_file = os.path.join(os.getcwd(), directory, 'canvas', f'brep_{largest_index}.step')
    return largest_step_file





def run_render_images(directory):
    # Load styles
    stroke_dataset_designer_name = 'Professional1'

    opacity_profiles_name = os.path.join(os.getcwd(), 'Preprocessing', 'proc_CAD','styles', 'opacity_profiles', stroke_dataset_designer_name+".json")
    if os.path.exists(opacity_profiles_name):
        with open(opacity_profiles_name, "r") as fp:
            opacity_profiles = json.load(fp)

    style_sheet_file_name = os.path.join(os.getcwd(), 'Preprocessing','proc_CAD', 'styles', 'stylesheets', stroke_dataset_designer_name+".json")
    if os.path.exists(style_sheet_file_name):
        with open(style_sheet_file_name, "r") as fp:
            stylesheet = json.load(fp)

    file_path = get_last_file(directory)
    edges_features = Preprocessing.proc_CAD.brep_read.create_graph_from_step_file(file_path)
    edges_features, obj_center= find_bounding_box(edges_features)
    optimize_opacities(edges_features, stylesheet)
    project_points(edges_features, obj_center)
    overshoot_stroke(edges_features)
    perturb_strokes(edges_features)

    plot_2d(edges_features, directory)