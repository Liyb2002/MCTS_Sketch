import numpy as np
import torch


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def chosen_face_id(boundary_points, edge_index_face_edge_list, index_id, edge_features):

    if edge_features.shape[1] == 1:
        boundary_points_tensor = torch.stack(boundary_points).to(torch.float32)
        return boundary_points_tensor

    # pair the edges index with each face
    face_to_edges = {}
    for face_edge_pair in edge_index_face_edge_list:
        face_list_index = face_edge_pair[0]
        edge_list_index = face_edge_pair[1]

        face_id = index_id[0][face_list_index].item()
        edge_id = index_id[0][edge_list_index].item()

        if face_id not in face_to_edges:
            face_to_edges[face_id] = []
        face_to_edges[face_id].append(edge_id)

    # builds which points a face have
    face_to_points = {}
    for face_id, edge_ids in face_to_edges.items():
        unique_points = set()
        for edge_id in edge_ids:
            # get the points for the edge
            edge_points = edge_features[0, edge_id, :]

            start_point = edge_points[:3]
            end_point = edge_points[3:]
            
            # add each point to the set
            unique_points.add(start_point)
            unique_points.add(end_point)

        # store the unique points in the dictionary
        face_to_points[face_id] = list(unique_points)


    # Find which face has all its point in the boundary_point
    # output is the face_id
    target_face_id = 0
    num_faces = len(face_to_points)

    boundary_points_values_set = {tuple(torch.cat(boundary_point).to(torch.float32).tolist()) for boundary_point in boundary_points}

    for face_id, face_points in face_to_points.items():
        face_points_values_set = {tuple(face_point.tolist()) for face_point in face_points}
        if boundary_points_values_set.issubset(face_points_values_set):
            target_face_id = face_id
            break
    
    # Now, build the ground truth matrix
    gt_matrix = torch.zeros((num_faces, 1))
    gt_matrix[target_face_id, 0] = 1

    return gt_matrix


def chosen_all_face_id(node_features, edge_index_face_edge_list, index_id, edge_features):


    # pair the edges index with each face
    face_to_edges = {}
    for face_edge_pair in edge_index_face_edge_list:
        face_list_index = face_edge_pair[0]
        edge_list_index = face_edge_pair[1]

        face_id = index_id[0][face_list_index].item()
        edge_id = index_id[0][edge_list_index].item()

        if face_id not in face_to_edges:
            face_to_edges[face_id] = []
        face_to_edges[face_id].append(edge_id)

    # builds which points a face have
    face_to_points = {}
    for face_id, edge_ids in face_to_edges.items():
        unique_points = set()
        for edge_id in edge_ids:
            # get the points for the edge
            edge_points = edge_features[0, edge_id, :]

            start_point = edge_points[:3]
            end_point = edge_points[3:]
            
            # add each point to the set
            unique_points.add(start_point)
            unique_points.add(end_point)

        # store the unique points in the dictionary
        face_to_points[face_id] = list(unique_points)


    # Find which face has all its point in the boundary_point
    # output is the face_id
    target_face_id = 0
    num_faces = len(face_to_points)

    satisfaction_matrix = []

    for _, face_points in face_to_points.items():
        satisfaction = check_face_satisfaction(face_points, node_features)
        satisfaction_matrix.append(satisfaction)

    satisfaction_matrix = torch.tensor(satisfaction_matrix, dtype=torch.float32)
    
    satisfaction_matrix = satisfaction_matrix.unsqueeze(1)
    return satisfaction_matrix


def identify_coplanar_direction(face_points):
    x_values = set(point[0].item() for point in face_points)
    y_values = set(point[1].item() for point in face_points)
    z_values = set(point[2].item() for point in face_points)

    if len(x_values) == 1:
        coplanar_direction = 'x'
        coplanar_value = next(iter(x_values))
    elif len(y_values) == 1:
        coplanar_direction = 'y'
        coplanar_value = next(iter(y_values))

    elif len(z_values) == 1:
        coplanar_direction = 'z'
        coplanar_value = next(iter(z_values))
    else:
        coplanar_direction = None
        coplanar_value = None

    return coplanar_direction, coplanar_value

def is_edge_within_bounds(edge, coplanar_direction, coplanar_value, face_points):
    point1 = edge[:3]
    point2 = edge[3:]

    if coplanar_direction == 'x':
        if point1[0] != coplanar_value or point2[0] != coplanar_value:
            return False
        y_values = [point[1] for point in face_points]
        z_values = [point[2] for point in face_points]
        y_min, y_max = min(y_values), max(y_values)
        z_min, z_max = min(z_values), max(z_values)
        return (y_min <= point1[1] <= y_max and y_min <= point2[1] <= y_max and
                z_min <= point1[2] <= z_max and z_min <= point2[2] <= z_max)
    elif coplanar_direction == 'y':
        if point1[1] != coplanar_value or point2[1] != coplanar_value:
            return False
        x_values = [point[0] for point in face_points]
        z_values = [point[2] for point in face_points]
        x_min, x_max = min(x_values), max(x_values)
        z_min, z_max = min(z_values), max(z_values)
        return (x_min <= point1[0] <= x_max and x_min <= point2[0] <= x_max and
                z_min <= point1[2] <= z_max and z_min <= point2[2] <= z_max)
    elif coplanar_direction == 'z':
        if point1[2] != coplanar_value or point2[2] != coplanar_value:
            return False
        x_values = [point[0] for point in face_points]
        y_values = [point[1] for point in face_points]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        return (x_min <= point1[0] <= x_max and x_min <= point2[0] <= x_max and
                y_min <= point1[1] <= y_max and y_min <= point2[1] <= y_max)
    else:
        return False

def check_face_satisfaction(face_points, node_features):
    
    coplanar_direction, coplanar_value = identify_coplanar_direction(face_points)
    if coplanar_direction is None:
        return 0  # Face is not coplanar in any single direction

    for edge in node_features:
        if is_edge_within_bounds(edge, coplanar_direction, coplanar_value, face_points):
            return 1  # Face is satisfied if at least one edge satisfies the conditions

    return 0  # Face is not satisfied






def find_left_edge(edge_features, node_features):
    # Extracting shapes
    n, _ = edge_features.shape
    m, _ = node_features.shape
    
    # Initialize the output tensor
    output = torch.ones((m, 1), dtype=torch.float32)
    
    # Iterate through each element in node_features
    for i in range(m):
        # Extract the current pair of 3D points from node_features
        node_pair = node_features[i, :]
        
        # Check if this pair is in any of the edge_features
        for j in range(n):
            if torch.equal(node_pair, edge_features[j, :]):
                output[i, 0] = 0
                break
    
    return output


def vis_gt_face(brep_edge_features, gt_matrix, edge_index_face_edge_list, index_id):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    num_edges = brep_edge_features.shape[1]

    face_to_edges = {}
    for face_edge_pair in edge_index_face_edge_list:
        face_list_index = face_edge_pair[0]
        edge_list_index = face_edge_pair[1]

        face_id = index_id[0][face_list_index].item()
        edge_id = index_id[0][edge_list_index].item()

        if face_id not in face_to_edges:
            face_to_edges[face_id] = []
        face_to_edges[face_id].append(edge_id)


    chosen_indices = (gt_matrix == 1).nonzero(as_tuple=True)[0].tolist()
    chosen_edges = []
    
    for index in chosen_indices:
        if index in face_to_edges:
            chosen_edges.extend(face_to_edges[index])
    


    for i in range(num_edges):
        start_point = brep_edge_features[0, i, :3]
        end_point = brep_edge_features[0, i, 3:]

        col = 'blue'
        if i in chosen_edges:
            col = 'red'

        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]], color=col)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()



def vis_predicted_face(brep_edge_features, predicted_index, edge_index_face_edge_list, index_id):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    num_edges = brep_edge_features.shape[1]

    face_to_edges = {}
    for face_edge_pair in edge_index_face_edge_list:
        face_list_index = face_edge_pair[0]
        edge_list_index = face_edge_pair[1]

        face_id = index_id[0][face_list_index].item()
        edge_id = index_id[0][edge_list_index].item()

        if face_id not in face_to_edges:
            face_to_edges[face_id] = []
        face_to_edges[face_id].append(edge_id)

    chosen_face = face_to_edges[predicted_index]


    for i in range(num_edges):
        start_point = brep_edge_features[0, i, :3]
        end_point = brep_edge_features[0, i, 3:]

        col = 'blue'
        if i in chosen_face:
            col = 'red'

        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]], color=col)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()



def vis_stroke_cloud(node_features):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    node_features = node_features.squeeze(0)

    # Plot all strokes in blue
    for stroke in node_features:
        start = stroke[:3].numpy()
        end = stroke[3:].numpy()
        
        # Plot the line segment for the stroke in blue
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], marker='o', color='blue')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def plot_strokes(strokes, color='b'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for stroke in strokes:
        x_values = [stroke[0], stroke[3]]
        y_values = [stroke[1], stroke[4]]
        z_values = [stroke[2], stroke[5]]

        ax.plot(x_values, y_values, z_values, color=color)

    # Set the limits of the axes to match the data range
    ax.set_xlim(strokes[:, [0, 3]].min(), strokes[:, [0, 3]].max())
    ax.set_ylim(strokes[:, [1, 4]].min(), strokes[:, [1, 4]].max())
    ax.set_zlim(strokes[:, [2, 5]].min(), strokes[:, [2, 5]].max())

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def vis_gt_strokes(brep_edge_features, gt_matrix):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    num_edges = brep_edge_features.shape[0]

    for i in range(num_edges):
        start_point = brep_edge_features[ i, :3]
        end_point = brep_edge_features[i, 3:]

        col = 'blue'
        if gt_matrix[i] > 0.5:
            col = 'red'

        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]], color=col)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()



def vis_predicted_strokes(brep_edge_features, predicted_matrix):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    predicted_matrix_np = predicted_matrix.detach().numpy().flatten()
    max4indices = np.argsort(predicted_matrix_np)[-4:]
    print("np.argsort(predicted_matrix_np)", np.argsort(predicted_matrix_np))


    num_edges = brep_edge_features.shape[1]

    for i in range(num_edges):
        start_point = brep_edge_features[0, i, :3]
        end_point = brep_edge_features[0, i, 3:]

        col = 'blue'
        if i in max4indices:
            col = 'red'

        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]], color=col)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def find_coplanar_axis(tensor):
    # Check if all x values are the same
    if torch.all(tensor[:, 0] == tensor[0, 0]):
        return 'x', tensor[0, 0].item()
    # Check if all y values are the same
    elif torch.all(tensor[:, 1] == tensor[0, 1]):
        return 'y', tensor[0, 1].item()
    # Check if all z values are the same
    elif torch.all(tensor[:, 2] == tensor[0, 2]):
        return 'z', tensor[0, 2].item()
    else:
        return None, None


def chosen_edge_id(boundary_points_tensor, edge_features):

    # Convert boundary_points to a tensor
    plane, value = find_coplanar_axis(boundary_points_tensor)
    
    # Initialize the output matrix
    num_edges = edge_features.shape[0]
    gt_matrix = torch.zeros((num_edges, 1), dtype=torch.float32)

    if plane is None:
        return None
     
    plane_idx = {'x': 0, 'y': 1, 'z': 2}[plane]

    # Loop through each edge
    for i in range(num_edges):
        start_point = edge_features[i, :3]
        end_point = edge_features[i, 3:]

        # Check if both the start and end points have the correct value in the plane
        if start_point[plane_idx] == value and end_point[plane_idx] == value:
            # Set the value in gt_matrix
            gt_matrix[i, 0] = 1
    
    return gt_matrix



def chosen_vertex_id(boundary_points, vertex_features):
    # Convert boundary_points to a tensor
    boundary_points_tensor = torch.tensor(boundary_points, dtype=torch.float32)
    
    # Initialize the output matrix
    num_verts = vertex_features.shape[0]
    gt_matrix = torch.zeros((num_verts), dtype=torch.float32)
    
    # Loop through each edge
    for i in range(num_verts):
        start_point = vertex_features[i, :3]

        # Check if both the start and end points are in boundary_points
        start_in_boundary = any(torch.equal(start_point, bp) for bp in boundary_points_tensor)
        
        # Set the value in gt_matrix
        if start_in_boundary:
            gt_matrix[i] = 1
    
    # print("gt_matrix", gt_matrix)
    return gt_matrix


def chosen_edge_id_stroke_cloud(boundary_points, node_features):

    boundary_points_list = [[float(point[0]), float(point[1]), float(point[2])] for point in boundary_points]
    
    # Initialize the output matrix
    num_edges = node_features.shape[1]
    gt_matrix = torch.zeros((num_edges, 1), dtype=torch.float32)
    
    # Loop through each edge
    for i in range(num_edges):
        start_point = node_features[0, i, :3]  # Assuming x, y, z are the first three dimensions
        end_point = node_features[0, i, 3:]    # Assuming x, y, z are the next three dimensions
        
        print("start_point", start_point, "end_point", end_point)
        # Check if start and end points are on the plane (exist in boundary_points)
        start_on_plane = (start_point[0] in boundary_points[0]) and (start_point[1] in boundary_points[1]) and (start_point[2] in boundary_points[2])
        end_on_plane = (end_point[0] in boundary_points[0]) and (end_point[1] in boundary_points[1]) and (end_point[2] in boundary_points[2])
        
        if start_on_plane and end_on_plane:
            gt_matrix[i] = 1.0  # Mark the edge as on the plane
        
    return gt_matrix



def edit_stroke_cloud(chosen_edges, node_features, operations_matrix, intersection_matrix, operations_order_matrix):
    # Convert chosen_edges to boolean mask based on >0.5 threshold
    chosen_mask = chosen_edges > 0.5
    chosen_mask = chosen_mask.squeeze(0)  # Remove the first dimension if present
    
    # Select indices of chosen edges
    chosen_indices = torch.nonzero(chosen_mask, as_tuple=False)[:, 0] 
    
    # Extract chosen node features
    node_features = node_features[:, chosen_indices, :]

    # Extract corresponding operations matrix
    operations_matrix = operations_matrix[:, chosen_indices, :]
    
    # Extract intersection matrix for chosen edges
    intersection_matrix = intersection_matrix[:, chosen_indices][:, :, chosen_indices]
    
    # Extract operations order matrix for chosen edges
    operations_order_matrix = operations_order_matrix[:, chosen_indices, :]
    
    return node_features, operations_matrix, intersection_matrix, operations_order_matrix



def coplanar_strokes(node_features, kth_operation):

    # get the chosen strokes
    chosen_strokes = []
    num_strokes = node_features.shape[0]
    
    for i in range(num_strokes):
        if kth_operation[i, 0] == 1:
            chosen_strokes.append(node_features[i, :])
    
    # for stroke in chosen_strokes:
    #     print("stroke", stroke)

    chosen_strokes = torch.stack(chosen_strokes)

    # Find the common plane
    common_values = {}
    planes = ['x', 'y', 'z']
    for plane_idx in range(3):
        # Get all values for the current plane (x, y, or z)
        plane_values = chosen_strokes[:, [plane_idx, plane_idx + 3]]
        # Flatten and find unique values
        unique_values = torch.unique(plane_values)
        for value in unique_values:
            if torch.sum(plane_values == value).item() == 2 * chosen_strokes.shape[0]:
                common_values[planes[plane_idx]] = value.item()
                break
    
    if len(common_values) == 0:
        return None
    common_plane = next(iter(common_values))
    common_value = common_values[common_plane]

    plane_idx = planes.index(common_plane)

    # Initialize coplanar matrix
    coplanar_matrix = torch.zeros( (num_strokes, 1), dtype=torch.float32)

    # Check all strokes in node_features for coplanarity
    for i in range(num_strokes):

        stroke = node_features[ i, :]
        point1, point2 = stroke[:3], stroke[3:]
        if point1[plane_idx] == common_value and point2[plane_idx] == common_value:
            coplanar_matrix[i] = 1.0
    
    return coplanar_matrix



def chosen_all_edge_id(node_features, edge_index_face_edge_list, index_id, edge_features):
    # pair the edges index with each face

    face_to_edges = {}
    for face_edge_pair in edge_index_face_edge_list:
        face_list_index = face_edge_pair[0]
        edge_list_index = face_edge_pair[1]

        face_id = index_id[0][face_list_index].item()
        edge_id = index_id[0][edge_list_index].item()

        if face_id not in face_to_edges:
            face_to_edges[face_id] = []
        face_to_edges[face_id].append(edge_id)

    # builds which points a face have
    face_to_points = {}
    for face_id, edge_ids in face_to_edges.items():
        unique_points = set()
        for edge_id in edge_ids:
            # get the points for the edge
            edge_points = edge_features[edge_id, :]

            start_point = edge_points[:3]
            end_point = edge_points[3:]
            
            # add each point to the set
            unique_points.add(start_point)
            unique_points.add(end_point)

        # store the unique points in the dictionary
        face_to_points[face_id] = list(unique_points)


    # Find which face has all its point in the boundary_point
    # output is the face_id
    target_face_id = 0
    num_faces = len(face_to_points)
    num_edges = edge_features.shape[0]

    satisfaction_matrix = []


    for _, face_points in face_to_points.items():
        satisfaction = check_face_satisfaction(face_points, node_features)
        satisfaction_matrix.append(satisfaction)

    chosen_edges_matrix = torch.zeros((num_edges, 1), dtype=torch.float32)
    
    # Iterate through satisfaction_matrix
    for face_id, is_chosen in enumerate(satisfaction_matrix):
        if is_chosen == 1:
            # Get the list of edge ids for the chosen face
            edges = face_to_edges.get(face_id, [])
            # Mark these edges as chosen
            for edge_id in edges:
                chosen_edges_matrix[edge_id] = 1
    
    return chosen_edges_matrix



def math_all_stroke_edges(node_features, edge_features):
    node_features = node_features.squeeze(0)
    edge_features = edge_features.squeeze(0)


    chosen_edges = torch.zeros((edge_features.shape[0], 1), dtype=torch.float32)
    
    # Iterate through each edge
    for i, edge in enumerate(edge_features):
        # Extract the points of the edge
        edge_start, edge_end = edge[:3], edge[3:]
        
        # Iterate through each node
        for node in node_features:
            # Extract the points of the node
            node_start, node_end = node[:3], node[3:]
            
            # Check for common value in x, y, z directions
            if (edge_start[0] == edge_end[0] == node_start[0] == node_end[0] or
                edge_start[1] == edge_end[1] == node_start[1] == node_end[1] or
                edge_start[2] == edge_end[2] == node_start[2] == node_end[2]):
                chosen_edges[i] = 1
                break  # No need to check other nodes if condition is met

    return chosen_edges


    
def node_features_to_plane(node_features):
    node_features = node_features.squeeze(0)

    x_values = set()
    y_values = set()
    z_values = set()

    # Iterate through each line in node_features
    for line in node_features:
        x1, y1, z1, x2, y2, z2 = line
        x_values.update([x1.item(), x2.item()])
        y_values.update([y1.item(), y2.item()])
        z_values.update([z1.item(), z2.item()])

    # Convert sets to sorted lists
    x_values = sorted(list(x_values))
    y_values = sorted(list(y_values))
    z_values = sorted(list(z_values))

    # Initialize lists to store plane indices
    x_planes = []
    y_planes = []
    z_planes = []

    # Iterate through unique values to create plane lists
    for x in x_values:
        plane = []
        for idx, line in enumerate(node_features):
            x1, y1, z1, x2, y2, z2 = line
            if x1.item() == x and x2.item() == x:
                plane.append(idx)
        if len(plane) >= 3:
            x_planes.append(plane)

    for y in y_values:
        plane = []
        for idx, line in enumerate(node_features):
            x1, y1, z1, x2, y2, z2 = line
            if y1.item() == y and y2.item() == y:
                plane.append(idx)
        if len(plane) >= 3:
            y_planes.append(plane)

    for z in z_values:
        plane = []
        for idx, line in enumerate(node_features):
            x1, y1, z1, x2, y2, z2 = line
            if z1.item() == z and z2.item() == z:
                plane.append(idx)
        if len(plane) >= 3:
            z_planes.append(plane)

    planes = x_planes + y_planes + z_planes

    return planes



def integrate_brep_probs(brep_edges_weights, brep_stroke_connection_matrix, stroke_coplanar_matrix):
    num_strokes = brep_stroke_connection_matrix.size(0)
    num_brep = brep_stroke_connection_matrix.size(1)

    # Initialize stroke weights to zero
    stroke_weights = torch.zeros(num_strokes, dtype=torch.float32)

    # Iterate over each brep edge
    for j in range(num_brep):
        connected_strokes = (brep_stroke_connection_matrix[:, j] == 1).nonzero().squeeze()
        connected_strokes = connected_strokes.unsqueeze(0)

        if connected_strokes.numel() == 0:  
            continue
        if len(connected_strokes.shape) > 1:
            connected_strokes = connected_strokes[0]
        for stroke in connected_strokes:
            for i in range(num_strokes):
                if stroke_coplanar_matrix[stroke, i] ==  1:
                    stroke_weights[i] += brep_edges_weights[j].item()

    return stroke_weights.view(-1, 1)



def choose_extrude_strokes(sketch_strokes, extrude_strokes_raw, node_features):
    # Initialize the output matrix with zeros
    num_strokes = extrude_strokes_raw.shape[0]
    extrude_strokes = torch.zeros((num_strokes, 1), dtype=torch.float32)
    
    # Iterate through all strokes in extrude_strokes_raw
    for i in range(num_strokes):
        if extrude_strokes_raw[i] == 1 and sketch_strokes[i] == 0:
            chosen_stroke_points = node_features[i]
            chosen = False

            # Check if any of the points of the chosen stroke match with any sketch stroke points
            for j in range(num_strokes):
                if sketch_strokes[j] == 1:
                    sketch_stroke_points = node_features[j]

                    # Compare points of the chosen stroke with the points of the sketch strokes
                    if (chosen_stroke_points[:3].tolist() in [sketch_stroke_points[:3].tolist(), sketch_stroke_points[3:].tolist()] or
                        chosen_stroke_points[3:].tolist() in [sketch_stroke_points[:3].tolist(), sketch_stroke_points[3:].tolist()]):
                        chosen = True
                        break

            # If the chosen stroke has one of its points in any of the sketch strokes, mark it as chosen
            if chosen:
                extrude_strokes[i] = 1
    
    return extrude_strokes
    


def choose_extrude_opposite_face(sketch_strokes, extrude_strokes_raw, node_features):
    # Initialize the output matrix with zeros
    num_strokes = extrude_strokes_raw.shape[0]
    extrude_strokes = torch.zeros((num_strokes, 1), dtype=torch.float32)
    
    # Iterate through all strokes in extrude_strokes_raw
    for i in range(num_strokes):
        if extrude_strokes_raw[i] == 1 and sketch_strokes[i] == 0:
            chosen_stroke_points = node_features[i]
            chosen = False

            # Check if any of the points of the chosen stroke match with any sketch stroke points
            for j in range(num_strokes):
                if sketch_strokes[j] == 1:
                    sketch_stroke_points = node_features[j]

                    # Compare points of the chosen stroke with the points of the sketch strokes
                    if (chosen_stroke_points[:3].tolist() in [sketch_stroke_points[:3].tolist(), sketch_stroke_points[3:].tolist()] or
                        chosen_stroke_points[3:].tolist() in [sketch_stroke_points[:3].tolist(), sketch_stroke_points[3:].tolist()]):
                        chosen = True
                        break

            # If the chosen stroke has one of its points in any of the sketch strokes, mark it as chosen
            if not chosen:
                extrude_strokes[i] = 1
    
    return extrude_strokes
    