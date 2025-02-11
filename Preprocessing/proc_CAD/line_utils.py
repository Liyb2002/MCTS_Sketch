
from itertools import combinations
from Preprocessing.proc_CAD.basic_class import Face, Edge, Vertex

import random
import math
import numpy as np


# -------------------- Midpoint Lines -------------------- #

def midpoint_lines(edges):

    if len(edges) == 3:
        return []
    
    if len(edges) == 4:
        return rectangle_midpoint_lines(edges)

    return []


def triangle_midpoint_lines(edges):
    """
    Computes the midpoint of the third edge and creates a new edge that connects
    this midpoint to the vertex formed by the two edges of equal length in the triangle.
    """
    # Step 1: Extract idx1 from the ID of the first edge
    first_edge_id = edges[0].id
    idx1 = first_edge_id.split('_')[1]  # Extract the index after 'edge_'
    mp = "mp"  # Label for the midpoint

    # Step 2: Identify the two edges with the same length and the third edge
    edge_lengths = [(edge, ((edge.vertices[0].position[0] - edge.vertices[1].position[0])**2 +
                            (edge.vertices[0].position[1] - edge.vertices[1].position[1])**2 +
                            (edge.vertices[0].position[2] - edge.vertices[1].position[2])**2) ** 0.5) 
                    for edge in edges]

    # Sort edges based on their lengths
    edge_lengths.sort(key=lambda x: x[1])

    # Two edges with equal length
    edge1, edge2 = edge_lengths[0][0], edge_lengths[1][0]

    # Third edge
    third_edge = edge_lengths[2][0]

    # Step 3: Find the midpoint of the third edge
    point1 = third_edge.vertices[0].position
    point2 = third_edge.vertices[1].position
    midpoint_coords = tuple((point1[i] + point2[i]) / 2 for i in range(3))

    # Create a new vertex for the midpoint
    vertex_id = f'vert_{idx1}_{mp}_0'
    midpoint_vertex = Vertex(id=vertex_id, position=midpoint_coords)

    # Step 4: Determine the shared vertex of the two equal-length edges
    common_vertex = None
    for v1 in edge1.vertices:
        if v1 in edge2.vertices:
            common_vertex = v1
            break

    # Step 5: Create a new edge connecting the midpoint to the common vertex
    edge_id = f"edge_{idx1}_{mp}_0"
    new_edge = Edge(id=edge_id, vertices=(midpoint_vertex, common_vertex))

    # Return the new edge
    return [new_edge]


def rectangle_midpoint_lines(edges):
    """
    Computes the midpoints of the edges and creates new edge objects that connect these midpoints
    in a manner parallel to the rectangle's edges.
    """

    # Step 1: Extract idx1 from the ID of the first edge
    first_edge_id = edges[0].id
    idx1 = first_edge_id.split('_')[1]  # Extract the index after 'edge_'
    mp = "mp"  # Label for the midpoint

    # Step 2: Calculate midpoints of the edges
    midpoints = []
    current_vertex_idx = 0  # Start the new vertex index from 0

    for edge in edges:
        # Extract the 3D positions of the two vertices of the edge
        point1 = edge.vertices[0].position
        point2 = edge.vertices[1].position

        # Calculate the midpoint of the edge
        midpoint_coords = tuple((point1[i] + point2[i]) / 2 for i in range(3))
        
        # Create a new vertex for the midpoint
        vertex_id = f'vert_{idx1}_{mp}_{current_vertex_idx}'
        midpoint_vertex = Vertex(id=vertex_id, position=midpoint_coords)
        midpoints.append(midpoint_vertex)
        current_vertex_idx += 1  # Increment vertex index

    # Step 3: Find lines connecting midpoints that differ in exactly one coordinate
    midpoint_edges = []
    current_edge_idx = 0  # Start the new edge index from 0

    for mp1, mp2 in combinations(midpoints, 2):
        # Check if the two midpoints differ in exactly one coordinate
        differences = sum(1 for i in range(3) if mp1.position[i] != mp2.position[i])
        if differences == 1:  # Differ in exactly one coordinate
            # Create a new edge object
            edge_id = f"edge_{idx1}_{mp}_{current_edge_idx}"
            new_edge = Edge(id=edge_id, vertices=(mp1, mp2))
            midpoint_edges.append(new_edge)
            current_edge_idx += 1  # Increment the edge index for the next edge

    return midpoint_edges



# -------------------- Diagonal Lines -------------------- #
def diagonal_lines(edges):

    """
    Creates diagonal lines by connecting the diagonal vertices of a rectangle formed by the given edges.
    """

    if len(edges) == 3:
        return []

    # Step 1: Extract idx1 from the ID of the first edge
    first_edge_id = edges[0].id
    idx1 = first_edge_id.split('_')[1]  # Extract the index after 'edge_'
    label = "diagonal"  # Label for diagonal lines

    # Step 2: Collect all unique vertices
    vertices = set()
    for edge in edges:
        vertices.update(edge.vertices)

    vertices = list(vertices)  # Convert the set back to a list for indexing

    # Step 3: Find diagonals
    diagonal_edges = []
    current_edge_idx = 0  # Start the new edge index from 0

    # Use the first vertex and find a diagonal vertex not on the same edge
    v1 = vertices[0]
    # Find a vertex not on the same edge as v1
    for v2 in vertices[1:]:
        if not any(v1 in edge.vertices and v2 in edge.vertices for edge in edges):
            # v1 and v2 are diagonal vertices
            edge_id = f"edge_{idx1}_{label}_{current_edge_idx}"
            new_edge = Edge(id=edge_id, vertices=(v1, v2))
            diagonal_edges.append(new_edge)
            current_edge_idx += 1
            break

    # The remaining two vertices form the other diagonal
    remaining_vertices = [v for v in vertices if v not in (v1, v2)]
    if len(remaining_vertices) == 2:
        v3, v4 = remaining_vertices
        edge_id = f"edge_{idx1}_{label}_{current_edge_idx}"
        new_edge = Edge(id=edge_id, vertices=(v3, v4))
        diagonal_edges.append(new_edge)

    return diagonal_edges



# -------------------- Projection Lines -------------------- #
def projection_lines(edges):
    """
    Computes the midpoints of the extruding lines (edges) and connects these midpoints 
    to form a sketch that is parallel to the original sketch.
    """
    # Step 1: Extract idx1 from the ID of the first edge
    first_edge_id = edges[0].id
    idx1 = first_edge_id.split('_')[1]  # Extract the index after 'edge_'
    mp = "projection"

    # Determine the number of edges to consider (first half)
    half_length = len(edges) // 2

    # Step 2: Calculate midpoints of the first half of the extruding lines
    midpoints = []
    current_vertex_idx = 0  # Start the new vertex index from 0

    for i in range(half_length, len(edges)):
        edge = edges[i]
        
        # Extract the 3D positions of the two vertices of the edge
        point1 = edge.vertices[0].position
        point2 = edge.vertices[1].position

        # Calculate the midpoint of the edge
        midpoint_coords = tuple((point1[i] + point2[i]) / 2 for i in range(3))

        # Create a new vertex for the midpoint
        vertex_id = f'vert_{idx1}_{mp}_{current_vertex_idx}'
        midpoint_vertex = Vertex(id=vertex_id, position=midpoint_coords)
        midpoints.append(midpoint_vertex)
        current_vertex_idx += 1  # Increment vertex index

    # Step 3: Connect the midpoints to form the new sketch
    midpoint_edges = []
    current_edge_idx = 0  # Start the new edge index from 0

    # Assuming the midpoints should be connected in sequence
    for i in range(len(midpoints)):
        mp1 = midpoints[i]
        mp2 = midpoints[(i + 1) % len(midpoints)]  # Next midpoint, wrapping around

        # Create a new edge object connecting adjacent midpoints
        edge_id = f"edge_{idx1}_{mp}_{current_edge_idx}"
        new_edge = Edge(id=edge_id, vertices=(mp1, mp2))
        midpoint_edges.append(new_edge)
        current_edge_idx += 1  # Increment the edge index for the next edge

    return midpoint_edges



# -------------------- bounding_box Lines -------------------- #

def bounding_box_lines(edges):
    """
    Creates a bounding box around the given edges. The bounding box is defined by 12 lines 
    connecting the minimum and maximum x, y, and z coordinates. Returns two things:
    1) All the bounding box edges, regardless of whether they exist in existing_edges or not.
    2) Only the new edges that were not in the existing edges.
    """

    if len(edges) != 6:
        return []

    # Initialize min and max values
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')

    # Step 1: Find the min and max x, y, z values
    for edge in edges:
        for vertex in edge.vertices:
            pos = vertex.position
            min_x, max_x = min(min_x, pos[0]), max(max_x, pos[0])
            min_y, max_y = min(min_y, pos[1]), max(max_y, pos[1])
            min_z, max_z = min(min_z, pos[2]), max(max_z, pos[2])

    # Define the 8 vertices of the bounding box
    bbox_vertices = [
        (min_x, min_y, min_z), (min_x, min_y, max_z),
        (min_x, max_y, min_z), (min_x, max_y, max_z),
        (max_x, min_y, min_z), (max_x, min_y, max_z),
        (max_x, max_y, min_z), (max_x, max_y, max_z)
    ]

    # Generate all 12 edges of the bounding box
    bbox_edges_indices = [
        (0, 1), (1, 3), (3, 2), (2, 0),  # Bottom face
        (4, 5), (5, 7), (7, 6), (6, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]

    # Convert the edges to a set of tuples for quick comparison
    existing_edges = set()
    for edge in edges:
        vertex_positions = tuple(sorted((tuple(edge.vertices[0].position), tuple(edge.vertices[1].position))))
        existing_edges.add(vertex_positions)

    # Step 2: Create the edges of the bounding box
    new_bounding_box_edges = []  # To store only the new edges
    current_edge_idx = 0  # Index for new edges

    for v1_idx, v2_idx in bbox_edges_indices:
        v1, v2 = bbox_vertices[v1_idx], bbox_vertices[v2_idx]
        vertex_positions = tuple(sorted((v1, v2)))

        # Create two new vertices
        vertex1 = Vertex(id=f'vert_bbox_{current_edge_idx}_0', position=v1)
        vertex2 = Vertex(id=f'vert_bbox_{current_edge_idx}_1', position=v2)

        # Create a new edge for this bounding box line
        edge_id = f'edge_bbox_{current_edge_idx}'
        new_edge = Edge(id=edge_id, vertices=(vertex1, vertex2))

        # Add only the new edges that do not already exist
        if vertex_positions not in existing_edges:
            new_bounding_box_edges.append(new_edge)
        
        current_edge_idx += 1

    return new_bounding_box_edges



# -------------------- Grid Lines -------------------- #
def grid_lines(prev_edges, edges):
    """
    Creates grid lines by extending the edges of the bounding box defined by the input edges.
    1) Finds the min and max values in the x, y, and z axes for the input edges to define a bounding box.
    2) Extends each edge of the bounding box to the nearest intersection with any edge in prev_edges.
    Returns a list of Edge objects.
    """
    # Step 1: Find the min and max values in the x, y, and z axes for the input edges
    x_min = y_min = z_min = float('inf')
    x_max = y_max = z_max = float('-inf')

    for edge in edges:
        for vertex in edge.vertices:
            pos = vertex.position
            x_min, x_max = min(x_min, pos[0]), max(x_max, pos[0])
            y_min, y_max = min(y_min, pos[1]), max(y_max, pos[1])
            z_min, z_max = min(z_min, pos[2]), max(z_max, pos[2])

    # Step 2: Define the 8 vertices of the bounding box
    bbox_vertices = [
        (x_min, y_min, z_min), (x_min, y_min, z_max),
        (x_min, y_max, z_min), (x_min, y_max, z_max),
        (x_max, y_min, z_min), (x_max, y_min, z_max),
        (x_max, y_max, z_min), (x_max, y_max, z_max)
    ]

    # Define the 12 edges of the bounding box
    bbox_edges_indices = [
        (0, 1), (1, 3), (3, 2), (2, 0),  # Bottom face
        (4, 5), (5, 7), (7, 6), (6, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]

    grid_edges = []  # To store the new grid lines
    edge_count = 0  # Unique ID counter for new edges

    # Step 3: Extend each edge of the bounding box to the nearest intersection with any edge in prev_edges
    for v1_idx, v2_idx in bbox_edges_indices:
        vertex1 = bbox_vertices[v1_idx]
        vertex2 = bbox_vertices[v2_idx]

        # Determine which axis is changing
        different_axis = None
        for i in range(3):
            if vertex1[i] != vertex2[i]:
                different_axis = i
                break

        if different_axis is None:
            continue  # Should not happen, as all edges should have one varying axis

        # Initial values to extend the edge
        extend_neg = None  # For extending in the negative direction
        extend_pos = None  # For extending in the positive direction

        # Check for intersections with prev_edges
        for other_edge_id, other_edge in prev_edges.items():
            for vertex in other_edge.vertices:
                pos = vertex.position[different_axis]

                # Extend in the negative direction for min values
                if pos < min(vertex1[different_axis], vertex2[different_axis]):
                    if extend_neg is None or pos > extend_neg:
                        extend_neg = pos

                # Extend in the positive direction for max values
                if pos > max(vertex1[different_axis], vertex2[different_axis]):
                    if extend_pos is None or pos < extend_pos:
                        extend_pos = pos

        # Extend edges in the negative direction if applicable
        if extend_neg is not None:
            new_vertex1_pos = list(vertex1)
            new_vertex2_pos = list(vertex2)
            new_vertex1_pos[different_axis] = extend_neg
            new_vertex2_pos[different_axis] = extend_neg
            new_vertex1 = Vertex(id=f'grid_vert_{edge_count}_neg1', position=tuple(new_vertex1_pos))
            new_vertex2 = Vertex(id=f'grid_vert_{edge_count}_neg2', position=tuple(new_vertex2_pos))
            new_edge_neg = Edge(id=f'grid_line_{edge_count}_neg', vertices=(new_vertex1, new_vertex2))
            grid_edges.append(new_edge_neg)
            edge_count += 1

        # Extend edges in the positive direction if applicable
        if extend_pos is not None:
            new_vertex3_pos = list(vertex1)
            new_vertex4_pos = list(vertex2)
            new_vertex3_pos[different_axis] = extend_pos
            new_vertex4_pos[different_axis] = extend_pos
            new_vertex3 = Vertex(id=f'grid_vert_{edge_count}_pos1', position=tuple(new_vertex3_pos))
            new_vertex4 = Vertex(id=f'grid_vert_{edge_count}_pos2', position=tuple(new_vertex4_pos))
            new_edge_pos = Edge(id=f'grid_line_{edge_count}_pos', vertices=(new_vertex3, new_vertex4))
            grid_edges.append(new_edge_pos)
            edge_count += 1

    print('num', len(grid_edges))
    return grid_edges


# !!!------------!!! All Edges Operations !!!------------!!! #


# -------------------- whole_bounding_box Lines -------------------- #

def whole_bounding_box_lines(all_edges):
    """
    Creates a bounding box around all the edges provided in the set `all_edges`. 
    The bounding box is defined by 12 lines connecting the minimum and maximum x, y, and z coordinates. 
    If a line is already inside `all_edges`, it will not be recreated.
    """
    # Initialize min and max values
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')

    # Step 1: Find the min and max x, y, z values from all edges
    for other_edge_id, other_edge in all_edges.items():
        if other_edge.is_circle:
            continue

        for vertex in other_edge.vertices:
            pos = vertex.position
            min_x, max_x = min(min_x, pos[0]), max(max_x, pos[0])
            min_y, max_y = min(min_y, pos[1]), max(max_y, pos[1])
            min_z, max_z = min(min_z, pos[2]), max(max_z, pos[2])

    # Define the 8 vertices of the bounding box
    bbox_vertices = [
        (min_x, min_y, min_z), (min_x, min_y, max_z),
        (min_x, max_y, min_z), (min_x, max_y, max_z),
        (max_x, min_y, min_z), (max_x, min_y, max_z),
        (max_x, max_y, min_z), (max_x, max_y, max_z)
    ]

    # Generate all 12 edges of the bounding box
    bbox_edges_indices = [
        (0, 1), (1, 3), (3, 2), (2, 0),  # Bottom face
        (4, 5), (5, 7), (7, 6), (6, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]

    # Convert the edges to a set of tuples for quick comparison
    existing_edges = set()
    for other_edge_id, other_edge in all_edges.items():
        if other_edge.is_circle:
            continue
        vertex_positions = tuple(sorted((tuple(other_edge.vertices[0].position), tuple(other_edge.vertices[1].position))))
        existing_edges.add(vertex_positions)

    # Step 2: Create the edges of the bounding box
    bounding_box_edges = []
    current_edge_idx = 0  # Index for new edges
    for v1_idx, v2_idx in bbox_edges_indices:
        v1, v2 = bbox_vertices[v1_idx], bbox_vertices[v2_idx]
        vertex_positions = tuple(sorted((v1, v2)))

        # Check if the edge already exists
        if vertex_positions not in existing_edges:
            # Create two new vertices
            vertex1 = Vertex(id=f'whole_vert_bbox_{current_edge_idx}_0', position=v1)
            vertex2 = Vertex(id=f'whole_vert_bbox_{current_edge_idx}_1', position=v2)

            # Create a new edge connecting these vertices
            edge_id = f'whole_edge_bbox_{current_edge_idx}'
            new_edge = Edge(id=edge_id, vertices=(vertex1, vertex2))
            bounding_box_edges.append(new_edge)
            current_edge_idx += 1

    return bounding_box_edges


# -------------------- perturbing_lines Lines -------------------- #

def perturbing_lines(all_edges):
    """
    Extends each line in the set of edges by moving point A to a new point C
    and point B to a new point D, effectively extending the original line in 
    both directions.
    """
    for other_edge_id, other_edge in all_edges.items():
        # Extract the 3D positions of the two vertices (A and B) of the edge
        pointA = other_edge.vertices[0].position
        pointB = other_edge.vertices[1].position

        # Step 1: Calculate the direction from A to B
        direction = tuple(pointB[i] - pointA[i] for i in range(3))

        # Calculate the magnitude (length) of the direction vector
        magnitude = math.sqrt(sum(direction[i] ** 2 for i in range(3)))

        # Normalize the direction vector
        if magnitude == 0:
            continue  # Skip if the magnitude is zero (degenerate line)
        normalized_direction = tuple(direction[i] / magnitude for i in range(3))

        # Step 2: Extend the line by a random amount in both directions
        extend_length = random.uniform(0.03, 0.05)

        # Calculate new positions for C and D
        pointC = tuple(pointB[i] + normalized_direction[i] * extend_length for i in range(3))
        pointD = tuple(pointA[i] - normalized_direction[i] * extend_length for i in range(3))

        # Update the original edge vertices to new extended points
        other_edge.vertices[0].position = pointD  # Move A to D
        other_edge.vertices[1].position = pointC  # Move B to C

    return all_edges


# -------------------- remove duplicate Lines -------------------- #

def remove_duplicate_lines(all_edges):
    """
    Removes duplicate edges from all_edges.
    An edge is considered a duplicate if it has the same start and end points as another edge 
    (after rounding to 4 decimals) or if the points are in reversed order.
    Decides which line to remove based on the edge type:
    - Always remove 'construction_line' over 'maybe_feature_line'.
    - If both are 'maybe_feature_line' or both are 'construction_line', remove one of them.
    """
    # Helper function to round a 3D point to 4 decimals
    def round_point(point):
        return tuple(round(coord, 4) for coord in point)

    # Dictionary to keep track of unique edges and their chosen representative
    unique_edges = {}

    # Iterate through each edge in all_edges
    for edge_id, edge in list(all_edges.items()):
        if edge.is_circle:
            continue
        # Get the rounded positions of the vertices
        p1 = round_point(edge.vertices[0].position)
        p2 = round_point(edge.vertices[1].position)

        # Create a normalized edge representation (smallest first to account for reversed order)
        edge_key = tuple(sorted([p1, p2]))

        # If this normalized edge representation is not in unique_edges, add it
        if edge_key not in unique_edges:
            unique_edges[edge_key] = edge
        else:
            # If a duplicate is found, decide which edge to remove based on type
            existing_edge = unique_edges[edge_key]

            # Logic to determine which edge to keep:
            # Always prefer 'maybe_feature_line' over 'construction_line'
            if existing_edge.edge_type == 'construction_line' and edge.edge_type == 'maybe_feature_line':
                # Replace the existing 'construction_line' with 'maybe_feature_line'
                unique_edges[edge_key] = edge
                del all_edges[existing_edge.order_count]  # Remove the older 'construction_line'
            elif existing_edge.edge_type == 'maybe_feature_line' and edge.edge_type == 'construction_line':
                # Keep the existing 'maybe_feature_line' and remove the new 'construction_line'
                del all_edges[edge.order_count]
            else:
                # Both edges are of the same type; you can remove either one
                # For simplicity, we remove the current one
                del all_edges[edge.order_count]

    return all_edges


def remove_single_point(all_edges):
    """
    Removes edges from all_edges where both points of the edge are the same.
    An edge is considered junk if its start and end points are identical.
    """
    # Helper function to round a 3D point to 4 decimals
    def round_point(point):
        return tuple(round(coord, 4) for coord in point)

    # Iterate through each edge in all_edges
    for edge_id, edge in list(all_edges.items()):
        if edge.is_circle:
            continue

        # Get the rounded positions of the vertices
        p1 = round_point(edge.vertices[0].position)
        p2 = round_point(edge.vertices[1].position)

        # If the start and end points are the same, remove the edge
        if p1 == p2:
            del all_edges[edge_id]

    return all_edges


def random_remove_construction_lines(all_edges):
    """
    Removes edges with edge_type 'construction_line' with a 50% chance.
    """
    # Iterate through each edge in all_edges
    for edge_id, edge in list(all_edges.items()):
        # Check if the edge is a 'construction_line'
        if edge.edge_type == 'construction_line':
            # Use random to determine if the edge should be removed (30% chance)
            if random.random() < 0.3:
                del all_edges[edge_id]

    return all_edges
 

 #----------------------------------------------------------------------------------#

def find_circle_points(sketch_face_radius, sketch_face_center, sketch_face_normal):
    center = sketch_face_center
    normal = sketch_face_normal
    
    # Normalize the normal vector (though it's always [±1,0,0] or similar)
    norm_magnitude = (normal[0]**2 + normal[1]**2 + normal[2]**2)**0.5
    normal = [normal[0] / norm_magnitude, normal[1] / norm_magnitude, normal[2] / norm_magnitude]
    
    # Find the plane of the circle based on the normal
    if normal == [1, 0, 0] or normal == [-1, 0, 0]:
        # Circle lies in the yz-plane
        point1 = [center[0], center[1], center[2] + sketch_face_radius]
        point2 = [center[0], center[1], center[2] - sketch_face_radius]
        point3 = [center[0], center[1] + sketch_face_radius, center[2]]
        point4 = [center[0], center[1] - sketch_face_radius, center[2]]
        
    elif normal == [0, 1, 0] or normal == [0, -1, 0]:
        # Circle lies in the xz-plane
        point1 = [center[0] + sketch_face_radius, center[1], center[2]]
        point2 = [center[0] - sketch_face_radius, center[1], center[2]]
        point3 = [center[0], center[1], center[2] + sketch_face_radius]
        point4 = [center[0], center[1], center[2] - sketch_face_radius]
        
    elif normal == [0, 0, 1] or normal == [0, 0, -1]:
        # Circle lies in the xy-plane
        point1 = [center[0] + sketch_face_radius, center[1], center[2]]
        point2 = [center[0] - sketch_face_radius, center[1], center[2]]
        point3 = [center[0], center[1] + sketch_face_radius, center[2]]
        point4 = [center[0], center[1] - sketch_face_radius, center[2]]
        
    else:
        raise ValueError("The normal vector must have two zeros and one ±1.")
    
    return point1, point2, point3, point4


def create_vertex_nodes(sketch_face_radius, sketch_face_center, new_sketch_face_center, sketch_face_normal, base_id):
    # Get 4 points for the "low" circle and 4 points for the "high" circle
    v1_low, v2_low, v3_low, v4_low = find_circle_points(sketch_face_radius, sketch_face_center, sketch_face_normal)
    v1_high, v2_high, v3_high, v4_high = find_circle_points(sketch_face_radius, new_sketch_face_center, sketch_face_normal)

    # Generate unique IDs for each of the vertices
    v1_low_id = f"edge_{base_id}_v1_low"
    v2_low_id = f"edge_{base_id}_v2_low"
    v3_low_id = f"edge_{base_id}_v3_low"
    v4_low_id = f"edge_{base_id}_v4_low"
    
    v1_high_id = f"edge_{base_id}_v1_high"
    v2_high_id = f"edge_{base_id}_v2_high"
    v3_high_id = f"edge_{base_id}_v3_high"
    v4_high_id = f"edge_{base_id}_v4_high"

    # Create Vertex objects for each point
    v1_low_vert = Vertex(id=v1_low_id, position=v1_low)
    v2_low_vert = Vertex(id=v2_low_id, position=v2_low)
    v3_low_vert = Vertex(id=v3_low_id, position=v3_low)
    v4_low_vert = Vertex(id=v4_low_id, position=v4_low)

    v1_high_vert = Vertex(id=v1_high_id, position=v1_high)
    v2_high_vert = Vertex(id=v2_high_id, position=v2_high)
    v3_high_vert = Vertex(id=v3_high_id, position=v3_high)
    v4_high_vert = Vertex(id=v4_high_id, position=v4_high)

    # Return the 8 vertices as a list (or a tuple, or another data structure if needed)
    return [v1_low_vert, v2_low_vert, v3_low_vert, v4_low_vert,
            v1_high_vert, v2_high_vert, v3_high_vert, v4_high_vert]



def create_edge_nodes(base_id, verts):
    # Assuming verts is a list of 8 vertices: [v1_low, v2_low, v3_low, v4_low, v1_high, v2_high, v3_high, v4_high]

    # Generate unique IDs for each of the edges
    e1_id = f"edge_{base_id}_e1"
    e2_id = f"edge_{base_id}_e2"
    e3_id = f"edge_{base_id}_e3"
    e4_id = f"edge_{base_id}_e4"

    # Create Edge objects for each pair of corresponding low and high vertices
    e1 = Edge(id=e1_id, vertices=[verts[0], verts[4]])  # Connect v1_low to v1_high
    e2 = Edge(id=e2_id, vertices=[verts[1], verts[5]])  # Connect v2_low to v2_high
    e3 = Edge(id=e3_id, vertices=[verts[2], verts[6]])  # Connect v3_low to v3_high
    e4 = Edge(id=e4_id, vertices=[verts[3], verts[7]])  # Connect v4_low to v4_high
    
    # Return the edges as a tuple
    return e1, e2, e3, e4


# -------------------- Fillet Lines -------------------- #
def edges_splited_by_fillet(target_verts, self_edges, self_vertices):
    fillet_feature_lines = []
    
    for vertex_data in target_verts:
        fillet_id = vertex_data['id']
        fillet_position = vertex_data['coordinates']

        for edge_id, edge in self_edges.items():
            if edge is None or edge.vertices is None or len(edge.vertices) != 2:
                continue
            position_1 = edge.vertices[0].position
            position_2 = edge.vertices[1].position

            if (position_1[0] == position_2[0] and position_1[1] == position_2[1] and position_1[2] != position_2[2]) or \
               (position_1[0] == position_2[0] and position_1[2] == position_2[2] and position_1[1] != position_2[1]) or \
               (position_1[1] == position_2[1] and position_1[2] == position_2[2] and position_1[0] != position_2[0]):

                if is_point_on_line(fillet_position, position_1, position_2):
                    # Create edge1 with vertices (position_1, fillet_position)
                    edge1 = Edge(
                        id=f"edge_1_{edge.id}",
                        vertices=[self_vertices[fillet_id], self_vertices[edge.vertices[0].id]]
                    )
                    # Create edge2 with vertices (position_2, fillet_position)
                    edge2 = Edge(
                        id=f"edge_2_{edge.id}",
                        vertices=[self_vertices[fillet_id], self_vertices[edge.vertices[1].id]]
                    )

                    # Append the new edges to the list
                    fillet_feature_lines.append(edge1)
                    fillet_feature_lines.append(edge2)

    unique_edges = []
    seen_edges = set()

    for edge in fillet_feature_lines:
        # Create a tuple of sorted vertex IDs to ensure undirected edge comparison
        edge_vertices = tuple(sorted([edge.vertices[0].id, edge.vertices[1].id]))
        
        # Check if this set of vertices has been seen before
        if edge_vertices not in seen_edges:
            unique_edges.append(edge)
            seen_edges.add(edge_vertices)

    return unique_edges



def is_point_on_line(point, line_start, line_end, tolerance=1e-6):
    """
    Checks if a point lies on a line segment between line_start and line_end
    by verifying if the point's x, y, z values are between the min and max values
    of the line segment's start and end points.
    """
    # Get the min and max values for x, y, z from the line segment

    if (np.allclose(point, line_start, atol=tolerance) or
        np.allclose(point, line_end, atol=tolerance)):
        return False

    min_x, max_x = min(line_start[0], line_end[0]), max(line_start[0], line_end[0])
    min_y, max_y = min(line_start[1], line_end[1]), max(line_start[1], line_end[1])
    min_z, max_z = min(line_start[2], line_end[2]), max(line_start[2], line_end[2])

    # Check if the point's x, y, z coordinates lie within the bounds
    if (min_x - tolerance <= point[0] <= max_x + tolerance and
        min_y - tolerance <= point[1] <= max_y + tolerance and
        min_z - tolerance <= point[2] <= max_z + tolerance):
        return True
    return False
