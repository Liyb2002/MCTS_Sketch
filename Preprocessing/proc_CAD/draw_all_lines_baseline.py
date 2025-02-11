import json
from Preprocessing.proc_CAD.basic_class import Face, Edge, Vertex
import Preprocessing.proc_CAD.line_utils

import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

import numpy as np
from scipy.interpolate import CubicSpline


class create_stroke_cloud():
    def __init__(self, directory, messy = False):
        self.directory = directory

        self.order_count = 0
        self.faces = {}
        self.edges = {}
        self.vertices = {}
        self.id_to_count = {}

        self.messy = messy

        self.load_file()    


        self.brep_directory = os.path.join(directory, 'canvas')
        self.brep_files = [file_name for file_name in os.listdir(self.brep_directory) if file_name.startswith('brep_') and file_name.endswith('.step')]
        self.brep_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))


    def load_file(self):
        program_path = os.path.join(self.directory, 'Program.json')
        with open(program_path, 'r') as file:
            self.data = json.load(file)

    


    def read_all(self):

        target_brep_file = self.brep_files[-1]
        brep_file_path = os.path.join(self.brep_directory, target_brep_file)
        self.brep_edges, _ = Preprocessing.SBGCN.brep_read.create_graph_from_step_file(brep_file_path)

        current_index = 0
        while current_index < len(self.data):
            op = self.data[current_index]
            self.parse_op(op, current_index)
            current_index += 1
            
            if op['operation'][0] == 'terminate':
                self.finishing_production()
                return   
        
        return


    def output(self, onlyStrokes = True):
        print("Outputting details of all components...")

        # Output vertices
        print("\nVertices:")
        if not onlyStrokes:
            for vertex_id, vertex in self.vertices.items():
                print(f"Vertex ID: {vertex_id}, Position: {vertex.position}")

            # Output faces
            print("\nFaces:")
            for face_id, face in self.faces.items():
                vertex_ids = [vertex.id for vertex in face.vertices]
                normal = face.normal
                print(f"Face ID: {face_id}, Vertices: {vertex_ids}, Normal: {normal}")


        # Output edges
        print("\nEdges:")
        for edge_id, edge in self.edges.items():
            vertex_ids = [vertex.id for vertex in edge.vertices]
            # Adding checks if 'Op' and 'order_count' are attributes of edge
            ops = getattr(edge, 'Op', 'No operations')
            order_count = getattr(edge, 'order_count', 'No order count')
            connected_edge_ids = getattr(edge, 'connected_edges', None)
        
            print(f"Edge ID: {edge_id}, Vertices: {vertex_ids},  Operations: {ops}, Order Count: {order_count}, Connected Edges: {connected_edge_ids}")


    def vis_stroke_cloud(self, directory, show=False, target_Op=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Remove grid and axes
        ax.grid(False)
        ax.set_axis_off()

        # Initialize min and max limits
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')
        z_min, z_max = float('inf'), float('-inf')

        for _, edge in self.edges.items():
            # Determine line color, alpha, and thickness based on edge type
            if edge.is_circle:
                radius = edge.radius 
                center = edge.center
                normal = edge.normal

                # Generate circle points in the XY plane
                theta = np.linspace(0, 2 * np.pi, 100)
                x_values = radius * np.cos(theta)
                y_values = radius * np.sin(theta)
                z_values = np.zeros_like(theta)  # Circle lies in the XY plane initially

                # Combine the coordinates into a matrix of shape (3, 100)
                circle_points = np.array([x_values, y_values, z_values])

                # Normalize the normal vector
                normal = normal / np.linalg.norm(normal)

                # Find rotation axis and angle to rotate the XY plane to align with the normal vector
                z_axis = np.array([0, 0, 1])  # Z-axis is the normal for the XY plane

                # If the normal vector is not already along the Z-axis, we calculate the rotation
                if not np.allclose(normal, z_axis):
                    rotation_axis = np.cross(z_axis, normal)
                    rotation_axis /= np.linalg.norm(rotation_axis)  # Normalize rotation axis
                    angle = np.arccos(np.dot(z_axis, normal))

                    # Create the rotation matrix using the rotation axis and angle (Rodrigues' rotation formula)
                    K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                                [rotation_axis[2], 0, -rotation_axis[0]],
                                [-rotation_axis[1], rotation_axis[0], 0]])

                    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

                    # Rotate the circle points
                    rotated_circle_points = np.dot(R, circle_points)
                else:
                    rotated_circle_points = circle_points  # No rotation needed if normal is already along Z-axis

                # Translate the circle to the center point
                if math.isnan(rotated_circle_points[0][0]):
                    rotated_circle_points = circle_points

                x_values = rotated_circle_points[0] + center[0]
                y_values = rotated_circle_points[1] + center[1]
                z_values = rotated_circle_points[2] + center[2]

                # Set random line thickness and alpha value
                line_thickness = np.random.uniform(0.7, 0.9)
                line_alpha_value = np.random.uniform(0.5, 0.8)

                # Update min and max limits
                x_min, x_max = min(x_min, x_values.min()), max(x_max, x_values.max())
                y_min, y_max = min(y_min, y_values.min()), max(y_max, y_values.max())
                z_min, z_max = min(z_min, z_values.min()), max(z_max, z_values.max())

                # Plot the circle
                ax.plot(x_values, y_values, z_values, color='black', alpha=line_alpha_value, linewidth=line_thickness)
                continue
                
            if edge.is_curve:
                start_point = np.array(edge.vertices[0].position)
                end_point = np.array(edge.vertices[1].position)
                center = edge.center

                radius = np.linalg.norm(start_point - center)

                # Determine the plane where the arc lies by checking which axis is constant
                shared_axes = np.isclose(start_point, center) & np.isclose(end_point, center)
                
                if np.sum(shared_axes) != 1:
                    raise ValueError("The arc points and center do not lie on a plane aligned with one of the axes.")

                # The axis where all points have the same value (constant axis)
                shared_axis = np.where(shared_axes)[0][0]  # This is the constant axis
                plane_axes = [axis for axis in range(3) if axis != shared_axis]

                # Calculate the angles for start_point and end_point relative to the center using atan2
                vector_start = np.array([start_point[plane_axes[0]], start_point[plane_axes[1]]]) - np.array([center[plane_axes[0]], center[plane_axes[1]]])
                vector_end = np.array([end_point[plane_axes[0]], end_point[plane_axes[1]]]) - np.array([center[plane_axes[0]], center[plane_axes[1]]])

                theta_start = np.arctan2(vector_start[1], vector_start[0])
                theta_end = np.arctan2(vector_end[1], vector_end[0])

                # Normalize angles to the range [-pi, pi]
                if theta_start < 0:
                    theta_start += 2 * np.pi
                if theta_end < 0:
                    theta_end += 2 * np.pi

                # Ensure that the difference is exactly 1.57 (quarter circle)
                angle_diff = theta_end - theta_start

                if np.abs(angle_diff) > np.pi / 2:  # If the difference exceeds 1.57 radians
                    # Find the correct direction for the arc (handle crossing the -pi to pi boundary)
                    if theta_start > theta_end:
                        # Go clockwise
                        theta_end = theta_start + np.pi / 2
                    else:
                        # Go counterclockwise
                        theta_end = theta_start - np.pi / 2

                # Generate angles for the arc (make sure it's no more than 1.57 radians apart)
                theta = np.linspace(theta_start, theta_end, 100)

                # Generate arc points using parametric circle equation on the plane
                arc_points = []
                for t in theta:
                    arc_x = center[plane_axes[0]] + radius * np.cos(t)
                    arc_y = center[plane_axes[1]] + radius * np.sin(t)
                    point = [0, 0, 0]  # Create a 3D point

                    # Set the shared axis (constant value for all points)
                    point[shared_axis] = center[shared_axis]
                    
                    # Assign the arc points to the correct axes
                    point[plane_axes[0]] = arc_x
                    point[plane_axes[1]] = arc_y
                    
                    arc_points.append(point)

                arc_points = np.array(arc_points)

                line_thickness = np.random.uniform(0.7, 0.9)
                line_alpha_value = np.random.uniform(0.5, 0.8)

                # Plot the arc points
                ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2], label='Arc', color='black', alpha=line_alpha_value, linewidth=line_thickness)
                continue

                
                


            if edge.edge_type == 'feature_line':
                line_color = 'black'
                line_alpha = edge.alpha_value
                line_thickness = np.random.uniform(0.7, 0.9)
            elif edge.edge_type == 'construction_line':
                line_color = 'black'
                line_alpha = edge.alpha_value
                line_thickness = np.random.uniform(0.4, 0.6)

            # Get edge points and perturb them to create a hand-drawn effect
            points = [vertex.position for vertex in edge.vertices]
            if len(points) == 2:
                # Original points
                x_values = np.array([points[0][0], points[1][0]])
                y_values = np.array([points[0][1], points[1][1]])
                z_values = np.array([points[0][2], points[1][2]])

                # Update min and max limits for each axis
                x_min, x_max = min(x_min, x_values.min()), max(x_max, x_values.max())
                y_min, y_max = min(y_min, y_values.min()), max(y_max, y_values.max())
                z_min, z_max = min(z_min, z_values.min()), max(z_max, z_values.max())

                # Add small random perturbations to make the line appear hand-drawn
                perturb_factor = 0.002  # Adjusted perturbation factor
                perturbations = np.random.normal(0, perturb_factor, (10, 3))  # 10 intermediate points

                # Create interpolated points for smoother curves
                t = np.linspace(0, 1, 10)  # Parameter for interpolation
                x_interpolated = np.linspace(x_values[0], x_values[1], 10) + perturbations[:, 0]
                y_interpolated = np.linspace(y_values[0], y_values[1], 10) + perturbations[:, 1]
                z_interpolated = np.linspace(z_values[0], z_values[1], 10) + perturbations[:, 2]

                # Use cubic splines to smooth the perturbed line
                cs_x = CubicSpline(t, x_interpolated)
                cs_y = CubicSpline(t, y_interpolated)
                cs_z = CubicSpline(t, z_interpolated)

                # Smooth curve points
                smooth_t = np.linspace(0, 1, 100)
                smooth_x = cs_x(smooth_t)
                smooth_y = cs_y(smooth_t)
                smooth_z = cs_z(smooth_t)

                # Plot edges with randomized line thickness and alpha
                ax.plot(smooth_x, smooth_y, smooth_z, color=line_color, alpha=line_alpha, linewidth=line_thickness)

        # Compute the center of the shape
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2

        # Compute the maximum difference across x, y, z directions
        max_diff = max(x_max - x_min, y_max - y_min, z_max - z_min)

        # Set the same limits for x, y, and z axes centered around the computed center
        ax.set_xlim([x_center - max_diff / 2, x_center + max_diff / 2])
        ax.set_ylim([y_center - max_diff / 2, y_center + max_diff / 2])
        ax.set_zlim([z_center - max_diff / 2, z_center + max_diff / 2])

        if show:
            plt.show()

        filepath = os.path.join(directory, '3d_visualization.png')
        plt.savefig(filepath)
        plt.close(fig)


    def vis_brep(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Initialize min and max limits for each axis
        x_min, x_max = np.inf, -np.inf
        y_min, y_max = np.inf, -np.inf
        z_min, z_max = np.inf, -np.inf

        # Plot all edges
        for edge in self.brep_edges:
            x_values = np.array([edge[0], edge[3]])
            y_values = np.array([edge[1], edge[4]])
            z_values = np.array([edge[2], edge[5]])

            # Plot the line in black
            ax.plot(x_values, y_values, z_values, color='black')

            # Update min and max limits for each axis
            x_min, x_max = min(x_min, x_values.min()), max(x_max, x_values.max())
            y_min, y_max = min(y_min, y_values.min()), max(y_max, y_values.max())
            z_min, z_max = min(z_min, z_values.min()), max(z_max, z_values.max())

        # Compute the center of the shape
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2

        # Compute the maximum difference across x, y, z directions
        max_diff = max(x_max - x_min, y_max - y_min, z_max - z_min)

        # Set the same limits for x, y, and z axes centered around the computed center
        ax.set_xlim([x_center - max_diff / 2, x_center + max_diff / 2])
        ax.set_ylim([y_center - max_diff / 2, y_center + max_diff / 2])
        ax.set_zlim([z_center - max_diff / 2, z_center + max_diff / 2])

        # Display the plot
        plt.show()

    
    def parse_op(self, Op, index):
        op = Op['operation'][0]

        if op == 'terminate':
            return

        # parse circle
        if len(Op['vertices']) == 0:
            self.parse_circle(Op, index)
            return

        for vertex_data in Op['vertices']:
            vertex = Vertex(id=vertex_data['id'], position=vertex_data['coordinates'])
            self.vertices[vertex.id] = vertex


        new_edges = []
        # fillet does not have edges. It has arcs
        if op == 'fillet':
            # arc_0
            arc_0_vertices = [self.vertices[v_id] for v_id in Op['operation'][5]['arc_0'][3]]
            arc_0 = Edge(id='arc0_' + Op['operation'][5]['arc_0'][3][0], vertices=arc_0_vertices)
            arc_0_center = Op['operation'][5]['arc_0'][2]
            arc_0.check_is_curve(arc_0_center)

            arc_0.set_Op(op, index)
            arc_0.set_order_count(self.order_count)
            self.order_count += 1
            new_edges.append(arc_0)

            # arc_1
            arc_1_vertices = [self.vertices[v_id] for v_id in Op['operation'][6]['arc_1'][3]]
            arc_1 = Edge(id='arc0_' + Op['operation'][6]['arc_1'][3][0], vertices=arc_1_vertices)
            arc_1_center = Op['operation'][6]['arc_1'][2]
            arc_1.check_is_curve(arc_1_center)

            arc_1.set_Op(op, index)
            arc_1.set_order_count(self.order_count)
            self.order_count += 1
            new_edges.append(arc_1)


        cur_op_vertex_ids = []
        for edge_data in Op['edges']:
            vertices = [self.vertices[v_id] for v_id in edge_data['vertices']]

            for v_id in edge_data['vertices']:
                cur_op_vertex_ids.append(v_id)

            edge = Edge(id=edge_data['id'], vertices=vertices)

            if op == 'chamfer':
                edge.set_edge_type('feature_line')

            edge.set_Op(op, index)
            edge.set_order_count(self.order_count)
            self.order_count += 1
            new_edges.append(edge)

            # self.edges[edge.order_count] = edge

        # Now add the new edges to self.edges
        self.add_new_edges(new_edges)

        construction_lines = []
        # Now, we need to generate the construction lines
        if op == 'sketch':
            construction_lines = Preprocessing.proc_CAD.line_utils.midpoint_lines(new_edges)
            construction_lines += Preprocessing.proc_CAD.line_utils.diagonal_lines(new_edges)                

        if op == 'extrude':
            construction_lines = Preprocessing.proc_CAD.line_utils.projection_lines(new_edges)
            construction_lines += Preprocessing.proc_CAD.line_utils.bounding_box_lines(new_edges)
            # construction_lines = Preprocessing.proc_CAD.line_utils.grid_lines(self.edges, new_edges)

        if op == 'fillet' or op == 'chamfer':
            fillet_vert_ids = []
            for vertex_data in Op['vertices']:
                fillet_vert_ids.append(vertex_data['id'])
            fillet_feature_lines = Preprocessing.proc_CAD.line_utils.edges_splited_by_fillet(Op['vertices'], self.edges, self.vertices)
            
            for line in fillet_feature_lines:
                line.set_edge_type('maybe_feature_line')
                line.set_order_count(self.order_count)
                self.order_count += 1
                self.edges[line.order_count] = line


        for line in construction_lines:
            line.set_edge_type('construction_line')
            line.set_order_count(self.order_count)
            self.order_count += 1
            self.edges[line.order_count] = line
        

        # find the edges that has the current operation 
        # but not created by the current operation
        # self.find_unwritten_edges(cur_op_vertex_ids, op, index)

        for face_data in Op['faces']:
            vertices = [self.vertices[v_id] for v_id in face_data['vertices']]
            normal = face_data['normal']
            face = Face(id=face_data['id'], vertices=vertices, normal=normal)
            self.faces[face.id] = face          


    def parse_circle(self, Op, index):
        if Op['operation'][0] == 'sketch':
            # circle sketch
            id = Op['faces'][0]['id']
            radius = Op['faces'][0]['radius']
            center = Op['faces'][0]['center']
            normal = Op['faces'][0]['normal']

            # Create a circle face
            circle_face = Face(id=id, vertices=[], normal=[])
            circle_face.check_is_circle(radius, center, normal)
            self.faces[circle_face.id] = circle_face  


            # Create a circle edge
            edge_id = f"edge_{len(self.edges)}_{id}"
            circle_edge = Edge(id=edge_id, vertices=None)
            circle_edge.check_is_circle(radius, center, normal)
            circle_edge.set_order_count(self.order_count)
            self.order_count += 1
            self.edges[circle_edge.order_count] = circle_edge
            circle_edge.set_Op(Op['operation'][0], index)


        
        if Op['operation'][0] == 'extrude':
            # cylinder extrude
            
            sketch_face_id = Op['operation'][1]
            extrude_amount = Op['operation'][2]
            sketch_face = self.faces[sketch_face_id]

            sketch_face_radius = sketch_face.radius
            sketch_face_center = sketch_face.center
            sketch_face_normal = sketch_face.normal

            new_sketch_face_normal = [-x for x in sketch_face.normal]
            extrusion = [x * extrude_amount for x in new_sketch_face_normal]
            new_sketch_face_center = [a + b for a, b in zip(sketch_face_center, extrusion)]
            new_sketch_face_id = Op['faces'][0]['id']


            # Create extrude edges

            verts = Preprocessing.proc_CAD.line_utils.create_vertex_nodes(sketch_face_radius, sketch_face_center, new_sketch_face_center, sketch_face_normal, len(self.vertices))
            for vert in verts:
                self.vertices[vert.id] = vert


            edges = Preprocessing.proc_CAD.line_utils.create_edge_nodes(len(self.edges), verts)

            for edge in edges:
                edge.set_order_count(self.order_count)
                edge.set_Op(Op['operation'][0], index)
                edge.set_edge_type('feature_line')
                self.order_count += 1
                self.edges[edge.order_count] = edge


            # Create a circle edge
            edge_id = f"edge_{len(self.edges)}_{new_sketch_face_id}_N"
            circle_edge = Edge(id=edge_id, vertices=None)
            circle_edge.check_is_circle(sketch_face_radius, new_sketch_face_center, new_sketch_face_normal)
            circle_edge.set_order_count(self.order_count)
            self.order_count += 1
            self.edges[circle_edge.order_count] = circle_edge
            circle_edge.set_Op(Op['operation'][0], index)


    def adj_edges(self):

        def vert_on_line(vertex, edge):
            # Get the two vertices of the edge
            v1, v2 = edge.vertices

            # Get positions of the vertices
            p1 = v1.position
            p2 = v2.position
            p3 = vertex.position

            # Check if the vertex is one of the line endpoints
            if p3 == p1 or p3 == p2:
                return True

            # Compute vectors
            vec1 = (p2[0] - p1[0], p2[1] - p1[1])
            vec2 = (p3[0] - p1[0], p3[1] - p1[1])

            # Check if vectors are collinear by cross product
            cross_product = vec1[0] * vec2[1] - vec1[1] * vec2[0]

            # If cross product is zero, the vectors are collinear (the vertex is on the line)
            if cross_product != 0:
                return False

            # Check if p3 is between p1 and p2 using the dot product
            dot_product = (p3[0] - p1[0]) * (p2[0] - p1[0]) + (p3[1] - p1[1]) * (p2[1] - p1[1])
            if dot_product < 0:
                return False

            squared_length = (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2
            if dot_product > squared_length:
                return False

            return True

        for edge_id, edge in self.edges.items():
            if edge.is_circle:
                continue

            connected_edge_ids = set()  

            for vertex in edge.vertices:
                for other_edge_id, other_edge in self.edges.items():

                    if other_edge.is_circle:
                        continue

                    if other_edge_id != edge_id and vert_on_line(vertex, other_edge):
                        connected_edge_ids.add(other_edge_id)
            
            edge.connected_edges = list(connected_edge_ids)

            # print(f"Edge {edge_id} is connected to edges: {list(connected_edge_ids)}")


    def find_unwritten_edges(self, cur_op_vertex_ids, op, index):
        vertex_id_set = set(cur_op_vertex_ids)

        for edge_id, edge in self.edges.items():
            if all(vertex.id in vertex_id_set for vertex in edge.vertices):
                edge.set_Op(op, index)

    
    def map_id_to_count(self):
        for edge_id, edge in self.edges.items():
            self.id_to_count[edge_id] = edge.order_count


    def add_new_edges(self, new_edges):
        """
        Adds new edges to the existing set of edges (self.edges).
        For each new edge:
        1) Checks if it is contained within any edge in self.edges.
        2) If not contained, adds it to self.edges.
        3) If contained, splits the existing edge and replaces it with the smallest possible edges.
        """

        # Helper function to determine if one edge is contained within another
        def is_contained(edge1, edge2):
            if (edge1.vertices is None) or (edge2.vertices is None): 
                return False
            
            """Check if edge2 (q1->q2) is contained within edge1 (p1->p2)."""
            p1, p2 = edge1.vertices[0].position, edge1.vertices[1].position
            q1, q2 = edge2.vertices[0].position, edge2.vertices[1].position

            # Round positions to 4 decimal places
            p1 = tuple(round(coord, 4) for coord in p1)
            p2 = tuple(round(coord, 4) for coord in p2)
            q1 = tuple(round(coord, 4) for coord in q1)
            q2 = tuple(round(coord, 4) for coord in q2)

            # Step 1: Calculate the direction vector of edge1
            direction = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
            direction_magnitude = (direction[0]**2 + direction[1]**2 + direction[2]**2) ** 0.5
            if direction_magnitude == 0:
                return False  # Degenerate edge

            # Normalize the direction vector
            unit_dir = (direction[0] / direction_magnitude, direction[1] / direction_magnitude, direction[2] / direction_magnitude)

            # Round the normalized direction vector
            unit_dir = tuple(round(coord, 4) for coord in unit_dir)

            # Check if q1 and q2 are on the line defined by edge1
            def is_point_on_line(p, p1, unit_dir):
                """Check if point p is on the line defined by point p1 and direction vector unit_dir."""
                t_values = []
                for i in range(3):
                    if unit_dir[i] != 0:  # Avoid division by zero
                        t = (p[i] - p1[i]) / unit_dir[i]
                        t_values.append(round(t, 4))  # Round t to 4 decimals

                return all(abs(t - t_values[0]) < 1e-6 for t in t_values)

            if not (is_point_on_line(q1, p1, unit_dir) and is_point_on_line(q2, p1, unit_dir)):
                return False  # q1 or q2 is not on the line defined by edge1

            # Check if q1 and q2 are between p1 and p2
            def is_between(p, p1, p2):
                """Check if point p is between points p1 and p2."""
                return all(min(round(p1[i], 4), round(p2[i], 4)) <= round(p[i], 4) <= max(round(p1[i], 4), round(p2[i], 4)) for i in range(3))

            return is_between(q1, p1, p2) and is_between(q2, p1, p2)

        # Helper function to create or reuse vertices
        def get_or_create_vertex(position, vertices_dict):
            """Returns an existing vertex if it matches the position or creates a new one."""
            position = tuple(round(coord, 4) for coord in position)  # Round position to 4 decimals
            for vertex in vertices_dict.values():
                if vertex.position == position:
                    return vertex
            vertex_id = f"vert_{len(vertices_dict)}"
            new_vertex = Vertex(id=vertex_id, position=position)
            vertices_dict[vertex_id] = new_vertex
            return new_vertex

        # Step 1: Iterate through each new edge
        for new_edge in new_edges:
            
            is_edge_contained = False
            edges_to_remove = []
            edges_to_add = []


            # Check if the new edge is contained within any existing edge
            for _, prev_edge in list(self.edges.items()):

                if (prev_edge is None )or (new_edge is None):
                    continue
                    
                if is_contained(prev_edge, new_edge):
                    # The new edge is contained within the previous edge
                    is_edge_contained = True

                    # Get positions of vertices
                    A, B = prev_edge.vertices[0].position, prev_edge.vertices[1].position
                    C, D = new_edge.vertices[0].position, new_edge.vertices[1].position

                    # Round positions to 4 decimal places
                    A = tuple(round(coord, 4) for coord in A)
                    B = tuple(round(coord, 4) for coord in B)
                    C = tuple(round(coord, 4) for coord in C)
                    D = tuple(round(coord, 4) for coord in D)

                    # Step 1: Find unique points and their order along the line
                    unique_points = {tuple(A): 'A', tuple(B): 'B', tuple(C): 'C', tuple(D): 'D'}
                    unique_positions = sorted(unique_points.keys(), key=lambda p: (p[0], p[1], p[2]))

                    # Step 2: Create or reuse vertices
                    vertex_map = {p: get_or_create_vertex(p, self.vertices) for p in unique_positions}

                    # Step 3: Create new edges for each consecutive pair of unique points
                    for i in range(len(unique_positions) - 1):
                        start = vertex_map[unique_positions[i]]
                        end = vertex_map[unique_positions[i + 1]]

                        # Skip adding if this segment is the same as the new_edge
                        if (start.position == C and end.position == D) or (start.position == D and end.position == C):
                            continue

                        edge_id = f"edge_{len(self.edges)}_{i}"
                        split_edge = Edge(id=edge_id, vertices=(start, end))
                        edges_to_add.append(split_edge)

                    edges_to_remove.append(prev_edge.order_count)
                    break  # No need to check other previous edges since it is already contained

            # Step 2: Add the new edge if not contained within any existing edge
            if not is_edge_contained:
                new_edge.set_order_count(self.order_count)
                self.order_count += 1
                self.edges[new_edge.order_count] = new_edge
            else:
                new_edge.set_order_count(self.order_count)
                self.order_count += 1
                self.edges[new_edge.order_count] = new_edge

                # Remove the contained edge and add the new split edges
                # for edge_id in edges_to_remove:
                #     self.edges[edge_id].set_edge_type('construction_line')
                #     del self.edges[edge_id]
                for edge in edges_to_add:
                    edge.set_order_count(self.order_count)
                    # edge.set_Op('na', self.current_index)
                    self.order_count += 1
                    self.edges[edge.order_count] = edge
        

    def determine_edge_type(self):
        """
        Determines the type of each edge in self.edges.
        For each edge with type 'maybe_feature_line':
        1) Checks if it is contained within any brep_edge in self.brep_edges.
        2) If contained, sets its type to 'feature_line'.
        3) If not contained, sets its type to 'construction_line'.
        """
        # Helper function to round a 3D point to 4 decimals
        def round_point(point):
            return tuple(round(coord, 4) for coord in point)

        # Helper function to check if an edge is contained within a brep edge
        def is_contained_in_brep(edge, brep_edge):
            """Check if edge (with two vertices) is contained within brep_edge (a list of 6 values)."""
            p1, p2 = edge.vertices[0].position, edge.vertices[1].position
            q1, q2 = tuple(brep_edge[:3]), tuple(brep_edge[3:])

            # Round the points for comparison
            p1, p2 = round_point(p1), round_point(p2)
            q1, q2 = round_point(q1), round_point(q2)

            # Check if both vertices of edge are on the brep edge
            def is_between(p, a, b):
                """Check if point p is between points a and b."""
                return all(min(a[i], b[i]) <= p[i] <= max(a[i], b[i]) for i in range(3))

            # Ensure the condition that if p1 and p2 have the same value on any axis, then q1 and q2 must also
            for i in range(3):  # Loop over x, y, z axes
                if p1[i] == p2[i]:  # Check if p1 and p2 have the same value on this axis
                    if q1[i] != q2[i]:  # If q1 and q2 do not have the same value on this axis
                        return False  # The brep edge does not satisfy the condition

            # Check if edge is contained by brep_edge or has the same vertices
            if (p1 == q1 and p2 == q2) or (p1 == q2 and p2 == q1):
                return True  # Same vertices
            elif is_between(p1, q1, q2) and is_between(p2, q1, q2):
                return True  # Both points are on the brep edge
            else:
                return False

        # Step 1: Iterate through each edge in self.edges
        for edge in self.edges.values():
            if edge.is_circle:
                edge.set_edge_type('feature_line')
                continue

            # Only process edges with type 'maybe_feature_line'
            if edge.edge_type == 'maybe_feature_line':
                contained_in_brep = False

                # Step 2: Check if this edge is contained in any brep_edge
                for brep_edge in self.brep_edges:
                    if is_contained_in_brep(edge, brep_edge):
                        edge_start = edge.vertices[0].position
                        edge_end = edge.vertices[1].position
                        
                        # Extract coordinates from the brep_edge
                        brep_start = brep_edge[:3]
                        brep_end = brep_edge[3:]

                        contained_in_brep = True
                        break

                # Step 3: Set edge type based on containment
                if contained_in_brep:
                    edge.set_edge_type('feature_line')
                else:
                    edge.set_edge_type('construction_line')


    def finishing_production(self):

        if self.messy:
            self.edges = Preprocessing.proc_CAD.line_utils.random_remove_construction_lines(self.edges)

        construction_lines = Preprocessing.proc_CAD.line_utils.whole_bounding_box_lines(self.edges)
        for line in construction_lines:
            line.set_edge_type('construction_line')
            line.set_order_count(self.order_count)
            self.order_count += 1
            self.edges[line.order_count] = line
        
        self.edges = Preprocessing.proc_CAD.line_utils.remove_duplicate_lines(self.edges)
        self.edges = Preprocessing.proc_CAD.line_utils.remove_single_point(self.edges)

        self.determine_edge_type()
        
        for edge_id, edge in self.edges.items():
            edge.set_alpha_value()

        
        self.adj_edges()
        self.map_id_to_count()
        # self.vis_stroke_cloud(self.directory, True)
        # self.vis_brep()



def create_stroke_cloud_class(directory, messy = False):
    stroke_cloud_class = create_stroke_cloud(directory, messy)
    return stroke_cloud_class

