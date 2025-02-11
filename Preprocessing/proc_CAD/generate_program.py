

import json
import numpy as np
import Preprocessing.proc_CAD.helper
import random
import Preprocessing.proc_CAD.random_gen

import os
import math
from Preprocessing.proc_CAD.basic_class import Face, Edge, Vertex

class Brep:
    def __init__(self):
        self.Faces = []
        self.Edges = []
        self.Vertices = []

        self.op = []
        self.idx = 0
        
    
    def init_sketch_op(self):

        axis = np.random.choice(['x', 'y', 'z'])
        points, normal = Preprocessing.proc_CAD.random_gen.generate_random_rectangle(axis)
        
        if axis == 'x':
            boundary_points = ([1, 0, 0])
        elif axis == 'y':
            boundary_points = ([0, 1, 0])
        elif axis == 'z':
            boundary_points = ([0, 0, 1])

        self._sketch_op(points, normal, boundary_points)


    def _sketch_op(self, points, normal, boundary_points):
        vertex_list = []
        for i, point in enumerate(points):
            vertex_id = f"vertex_{self.idx}_{i}"
            vertex = Vertex(vertex_id, point.tolist())
            self.Vertices.append(vertex)
            vertex_list.append(vertex)

        num_vertices = len(vertex_list)
        for i in range(num_vertices):
            edge_id = f"edge_{self.idx}_{i}"
            edge = Edge(edge_id, [vertex_list[i], vertex_list[(i+1) % num_vertices]])  # Loop back to first vertex to close the shape
            edge.enable_fillet()
            self.Edges.append(edge)

        face_id = f"face_{self.idx}_{0}"
        face = Face(face_id, vertex_list, normal)
        face.face_fixed()
        self.Faces.append(face)
        
        self.idx += 1

        if type(boundary_points) != list:
            boundary_points = boundary_points.tolist()
        
        self.op.append(['sketch', boundary_points])


    def regular_sketch_op(self):
        self.face_validation_check()
        faces_with_future_sketch = [face for face in self.Faces if face.future_sketch ]
            
        if not faces_with_future_sketch:
            return False
        target_face = random.choice(faces_with_future_sketch)
        target_face.face_fixed()

        boundary_points = [vert.position for vert in target_face.vertices]
        normal = [ 0 - normal for normal in target_face.normal]

        cases = ['create_circle', 'find_rectangle', 'find_triangle', 'triangle_to_cut']
        # cases = ['create_circle']
        selected_case = random.choice(cases)
        if selected_case == 'create_circle':
            radius, center = Preprocessing.proc_CAD.helper.random_circle(boundary_points, normal)
            self.regular_sketch_circle(normal, radius, center)
            return 

        if selected_case == 'find_rectangle':
            random_polygon_points = Preprocessing.proc_CAD.helper.find_rectangle_on_plane(boundary_points, normal)

        if selected_case == 'find_triangle':
            random_polygon_points = Preprocessing.proc_CAD.helper.find_triangle_on_plane(boundary_points, normal)

        if selected_case == 'triangle_to_cut':
            random_polygon_points = Preprocessing.proc_CAD.helper.find_triangle_to_cut(boundary_points, normal)

        self._sketch_op(random_polygon_points, normal, boundary_points)


    def regular_sketch_circle(self, normal, radius, center):
        face_id = f"face_{self.idx}_{0}"
        face = Face(face_id, [], normal)
        face.check_is_circle(radius, center, normal)
        self.Faces.append(face)
        
        self.idx += 1
        self.op.append(['sketch'])


    def extrude_op(self, extrude_target_point = None):
        

        sketch_face = self.Faces[-1]
        new_vertices = []
        new_edges = []
        new_faces = []
        sketch_face_opposite_normal = [-x for x in sketch_face.normal]


        # For dataset generation process
        if extrude_target_point is None:
            amount = Preprocessing.proc_CAD.random_gen.generate_random_extrude()
            safe_amount = -self.safe_extrude_check()
            if amount <0:
                amount = max(amount, safe_amount)

            if self.idx < 2:
                amount = abs(amount)

            for i, vertex in enumerate(sketch_face.vertices):

                new_pos = [vertex.position[j] + sketch_face_opposite_normal[j] * amount for j in range(3)]
                vertex_id = f"vertex_{self.idx}_{i}"
                new_vertex = Vertex(vertex_id, new_pos)
                self.Vertices.append(new_vertex)
                new_vertices.append(new_vertex)

        else:
            extrude_direction = None
            amount = None

            for sketch_face_vert in sketch_face.vertices:
                vert_pos = sketch_face_vert.position
                # Check if vert_pos differs from extrude_target_point by only one axis
                diff_axis_values = [extrude_target_point[j] - vert_pos[j] for j in range(3)]
                diff_non_zero = [abs(val) > 1e-2 for val in diff_axis_values]
                
                if sum(diff_non_zero) == 1:  # Only one axis has a significant difference
                    extrude_direction = [1 if val > 0 else -1 if val < 0 else 0 for val in diff_axis_values]
                    amount = max(abs(val) for val in diff_axis_values).item()
                    break
            
            for i, sketch_face_vert in enumerate(sketch_face.vertices):
                vert_pos = sketch_face_vert.position
                # Map vert_pos to the same plane as extrude_target_point
                new_pos = [vert_pos[j] + extrude_direction[j] * amount for j in range(3)]
                vertex_id = f"vertex_{self.idx}_{i}"
                new_vertex = Vertex(vertex_id, new_pos)
                self.Vertices.append(new_vertex)
                new_vertices.append(new_vertex)


            if sketch_face.is_circle:
                center = sketch_face.center  # Circle's center

                # Compute the difference between the center and the extrude target point
                diff_axis_values = [extrude_target_point[j] - center[j] for j in range(3)]
                diff_non_zero = [abs(val) > 1e-2 for val in diff_axis_values]

                if sum(diff_non_zero) == 1:  # Only one axis has a significant difference
                    # Determine the extrusion direction and amount
                    normal = [1 if val > 0 else -1 if val < 0 else 0 for val in diff_axis_values]  # Extrusion direction (normal)
                    amount = max(abs(val) for val in diff_axis_values)  # Extrusion amount
                else:
                    raise ValueError("Extrusion target point does not differ from center by exactly one axis.")

                # The computed 'amount' and 'normal' can now be used for further operations
                print(f"Extrusion amount: {amount}, Normal: {normal}")


            if random.random() < 0.5 and len(self.op) > 2:
                amount = -amount


        num_vertices = len(new_vertices)
        for i in range(num_vertices):
            edge_id = f"edge_{self.idx}_{i}"
            edge = Edge(edge_id, [new_vertices[i], new_vertices[(i+1) % num_vertices]])  # Loop back to first vertex to close the shape
            edge.enable_fillet()
            self.Edges.append(edge)
            new_edges.append(edge)

        face_id = f"face_{self.idx}_{0}"
        new_face = Face(face_id, new_vertices, sketch_face_opposite_normal)
        self.Faces.append(new_face)
        new_faces.append(new_face)
        
        
        #create side edges and faces
        for i in range(num_vertices):
            # Vertical edges from old vertices to new vertices
            vertical_edge_id = f"edge_{self.idx}_{i+num_vertices}"
            vertical_edge = Edge(vertical_edge_id, [sketch_face.vertices[i], new_vertices[i]])
            vertical_edge.enable_fillet()
            self.Edges.append(vertical_edge)

            # Side faces formed between pairs of old and new vertices
            side_face_id = f"face_{self.idx}_{i}"
            side_face_vertices = [
                sketch_face.vertices[i], new_vertices[i],
                new_vertices[(i + 1) % num_vertices], sketch_face.vertices[(i + 1) % num_vertices]
            ]
            normal = Preprocessing.proc_CAD.helper.compute_normal(side_face_vertices, new_vertices[(i + 2) % num_vertices])
            side_face = Face(side_face_id, side_face_vertices, normal)
            self.Faces.append(side_face)

        self.idx += 1
        self.op.append(['extrude', sketch_face.id, amount])


    def random_fillet(self, target_edge_tensor = None, amount = 0):        

        available_fillet_edges = [edge for edge in self.Edges if edge.fillet_permited]
        if not available_fillet_edges:
            return False
        
        if target_edge_tensor is None:
            target_edge = None
            while not self.check_fillet_validity(target_edge):
                target_edge = random.choice(available_fillet_edges)
        else:
            # find target_edge based on target_edge_tensor value
            target_edge = self.find_target_edge(target_edge_tensor)


        if amount == 0:
            amount = Preprocessing.proc_CAD.random_gen.generate_random_fillet()
            safe_amount = self.safe_fillet_check([vert.position for vert in target_edge.vertices])
            amount = min(amount * 0.5, safe_amount * 0.5)

        
        target_edge.disable_fillet()

        verts_pos = []
        verts_id = []
        new_vert_pos = []
        centers = []

        for vert in target_edge.vertices:
            verts_pos.append(vert.position)
            verts_id.append(vert.id)
            neighbor_verts = Preprocessing.proc_CAD.helper.get_neighbor_verts(vert, target_edge, self.Edges)
            new_vert_pos_half, center = Preprocessing.proc_CAD.helper.compute_fillet_new_vert(vert, neighbor_verts, amount)
            new_vert_pos.append(new_vert_pos_half)
            centers.append(center)
        
        new_A = new_vert_pos[0][0]
        new_B = new_vert_pos[0][1]
        new_C = new_vert_pos[1][0]
        new_D = new_vert_pos[1][1]

        #create 4 new verts from new_A, new_B and new_C, new_D
        new_vert_B = Vertex(f"vertex_{self.idx}_0", new_B)
        new_vert_D = Vertex(f"vertex_{self.idx}_1", new_D)
        new_vert_A = Vertex(f"vertex_{self.idx}_2", new_A)
        new_vert_C = Vertex(f"vertex_{self.idx}_3", new_C)
        self.Vertices.append(new_vert_B)
        self.Vertices.append(new_vert_D)
        self.Vertices.append(new_vert_A)
        self.Vertices.append(new_vert_C)

        arc_0 = [new_A, new_B, centers[0], [new_vert_A.id, new_vert_B.id]]
        arc_1 = [new_C, new_D, centers[1], [new_vert_C.id, new_vert_D.id]]


        #create 2 edge that connect new_B and new_D / new_A and new_C
        new_edge_id_0 = f"edge_{self.idx}_0"
        new_edge_0 = Edge(new_edge_id_0, [new_vert_B, new_vert_D])
        new_edge_id_1 = f"edge_{self.idx}_1"
        new_edge_1 = Edge(new_edge_id_1, [new_vert_A, new_vert_C])
        self.Edges.append(new_edge_0)
        self.Edges.append(new_edge_1)

        # Fillet should produce arc connecting the fillet verts
        #create 2 edge that connect new_A and new_B / new_C and new_D
        # new_edge_id_2 = f"edge_{self.idx}_2"
        # new_edge_2 = Edge(new_edge_id_2, [new_vert_A, new_vert_B])
        # new_edge_id_3 = f"edge_{self.idx}_3"
        # new_edge_3 = Edge(new_edge_id_3, [new_vert_C, new_vert_D])
        # self.Edges.append(new_edge_2)
        # self.Edges.append(new_edge_3)

        self.idx += 1
        self.op.append(['fillet', 
                        target_edge.id, 
                        {'amount': amount}, 
                        {'old_verts_pos': verts_pos},
                        {'verts_id': verts_id},
                        {'arc_0': arc_0},
                        {'arc_1': arc_1},
                        ])
    

    def random_chamfer(self, target_edge_tensor = None, amount = 0):
        # Chamfer is just non-linear version of fillet
        # so we copy fillet code for it
        available_fillet_edges = [edge for edge in self.Edges if edge.fillet_permited]
        if not available_fillet_edges:
            return False
        

        if target_edge_tensor is None:
            target_edge = None
            while not self.check_fillet_validity(target_edge):
                target_edge = random.choice(available_fillet_edges)
        else:
            # find target_edge based on target_edge_tensor value
            target_edge = self.find_target_edge(target_edge_tensor)



        if amount == 0:
            amount = Preprocessing.proc_CAD.random_gen.generate_random_fillet()
            safe_amount = self.safe_fillet_check([vert.position for vert in target_edge.vertices])
            amount = min(amount * 0.5, safe_amount * 0.5)


        target_edge.disable_fillet()


        verts_pos = []
        verts_id = []
        new_vert_pos = []
        centers = []

        for vert in target_edge.vertices:
            verts_pos.append(vert.position)
            verts_id.append(vert.id)
            neighbor_verts = Preprocessing.proc_CAD.helper.get_neighbor_verts(vert, target_edge, self.Edges)
            new_vert_pos_half, center = Preprocessing.proc_CAD.helper.compute_fillet_new_vert(vert, neighbor_verts, amount)
            new_vert_pos.append(new_vert_pos_half)
            centers.append(center)
        
        new_A = new_vert_pos[0][0]
        new_B = new_vert_pos[0][1]
        new_C = new_vert_pos[1][0]
        new_D = new_vert_pos[1][1]

        #create 4 new verts from new_A, new_B and new_C, new_D
        new_vert_B = Vertex(f"vertex_{self.idx}_0", new_B)
        new_vert_D = Vertex(f"vertex_{self.idx}_1", new_D)
        new_vert_A = Vertex(f"vertex_{self.idx}_2", new_A)
        new_vert_C = Vertex(f"vertex_{self.idx}_3", new_C)
        self.Vertices.append(new_vert_B)
        self.Vertices.append(new_vert_D)
        self.Vertices.append(new_vert_A)
        self.Vertices.append(new_vert_C)



        #create 2 edge that connect new_B and new_D / new_A and new_C
        new_edge_id_0 = f"edge_{self.idx}_0"
        new_edge_0 = Edge(new_edge_id_0, [new_vert_B, new_vert_D])
        new_edge_id_1 = f"edge_{self.idx}_1"
        new_edge_1 = Edge(new_edge_id_1, [new_vert_A, new_vert_C])
        self.Edges.append(new_edge_0)
        self.Edges.append(new_edge_1)

        #create 2 edge that connect new_A and new_B / new_C and new_D
        new_edge_id_2 = f"edge_{self.idx}_2"
        new_edge_2 = Edge(new_edge_id_2, [new_vert_A, new_vert_B])
        new_edge_id_3 = f"edge_{self.idx}_3"
        new_edge_3 = Edge(new_edge_id_3, [new_vert_C, new_vert_D])
        self.Edges.append(new_edge_2)
        self.Edges.append(new_edge_3)

        self.idx += 1
        self.op.append(['chamfer', 
                        target_edge.id, 
                        {'amount': amount}, 
                        {'old_verts_pos': verts_pos},
                        {'verts_id': verts_id},
                        ])


    def check_fillet_validity(self, target_edge):
        if target_edge is None:
            return False
        
        for vert in target_edge.vertices:
            neighbor_verts = Preprocessing.proc_CAD.helper.get_neighbor_verts(vert, target_edge, self.Edges)

            if len(neighbor_verts) != 2:
                return False
        
        return True


    def write_to_json(self, data_directory = None):
        #clean everything in the folder
        folder = os.path.join(os.path.dirname(__file__), 'canvas')
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        if data_directory:
            folder = data_directory


        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

        #start writing program
        filename = os.path.join(folder, 'Program.json')
        data = []
        for count in range(0, self.idx):
            op = self.op[count][0]
            self.write_Op(self.op[count], count, data)
        
        self.write_terminate(data)  

        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"Data saved to {filename}")


    def write_Op(self, Op, index, data):
        operation = {
            'operation': Op,
            'faces': [],
            'edges': [],
            'vertices': []
        }

        # Add each point with an ID to the vertices list
        
        for face in self.Faces:
            if face.id.split('_')[1] == str(index):

                if face.is_circle:
                    face = {
                    'id': face.id,
                    'radius': face.radius,
                    'center': [pt for pt in face.center],
                    'normal': [float(n) if isinstance(n, np.floating) else int(n) for n in face.normal]
                    }
                else:
                    face = {
                        'id': face.id,
                        'vertices': [vertex.id for vertex in face.vertices],
                        'normal': [float(n) if isinstance(n, np.floating) else int(n) for n in face.normal]
                    }

                operation['faces'].append(face)

        for edge in self.Edges:
            if edge.id.split('_')[1] == str(index):
                
                edge = {
                    'id': edge.id,
                    'vertices': [vertex.id for vertex in edge.vertices]
                }
                operation['edges'].append(edge)


        
        for vertex in self.Vertices:
            if vertex.id.split('_')[1] == str(index):
                vertex = {
                    'id': vertex.id,
                    'coordinates': vertex.position 
                }
                operation['vertices'].append(vertex)
        
        data.append(operation)

        return data
                

    def write_terminate(self, data):
        operation = {
            'operation': ['terminate']
        }
        data.append(operation)
        return data


    def face_validation_check(self):
        checked_faces = set()

        for face in self.Faces:
            current_plane = face.plane
            
            for other_face in self.Faces:
                if other_face is face:
                    continue
                
                if other_face.plane == current_plane and other_face not in checked_faces:
                    face.face_fixed()
                    other_face.face_fixed()
            
            checked_faces.add(face)


    def safe_extrude_check(self):
        sketch_face = self.Faces[-1]
        sketch_plane = sketch_face.plane  # tuple, e.g., (x, 0) or (y, 0) or (z, 0)

        extrude_directions = [x for x in sketch_face.normal]


        extrude_direction = -1
        if 1 in extrude_directions:
            extrude_direction = 1
        
        base_value = sketch_plane[1]  # The value on the axis to compare against

        closest_value = float('inf')
        inf_big = float('inf')

        for face in self.Faces:
            if face is sketch_face:
                continue
            
            other_plane = face.plane
            distance = 0
            # Check if the face is on the same axis (x, y, or z)
            if other_plane[0] == sketch_plane[0]:
                other_value = other_plane[1]

                # Postive extrude
                if extrude_direction > 0 and other_value > base_value:
                    distance = abs(other_value - base_value)

                # Negative extrude
                if extrude_direction < 0 and other_value < base_value:
                    distance = abs(other_value - base_value)

                # We want the minimum distance
                if distance > 0 and distance < closest_value:
                    closest_value = distance

        # Return the minimum distance or inf_big if no valid face was found
        if closest_value == float('inf'):
            return inf_big
        else:
            return closest_value * 0.9


    def safe_fillet_check(self, points):
        """
        Finds the minimum distance between edge vertices in the provided points and sets max_dist accordingly.
        """

        def euclidean_dist(p1, p2):
            """
            Computes the Euclidean distance between two 3D points.
            """
            return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

        max_dist = float('inf')  # Set to a very large number
        for e in self.Edges:
            if not e.is_circle:  # Skip circular edges
                for vert in e.vertices:
                    vert_pos = vert.position
                    if vert_pos in points:  # Check if vertex position is in the provided points
                        dist = max(euclidean_dist(e.vertices[0].position, vert_pos), euclidean_dist(e.vertices[1].position, vert_pos))
                        if dist < max_dist:
                            max_dist = dist

        return max_dist if max_dist != float('inf') else None  # Return None if no valid distance was found
    

    def find_target_edge(self, edge_tensor):
        point_1 = edge_tensor[:3].tolist()
        point_2 = edge_tensor[3:6].tolist()

        def is_close(p1, p2, tol=1e-6):
            return all(abs(a - b) < tol for a, b in zip(p1, p2))

        for edge in self.Edges:
            pos_1 = edge.vertices[0].position
            pos_2 = edge.vertices[1].position

            # print("point_1", point_1, "point_2", point_2)
            # print("pos_1", pos_1, "pos_2", pos_2)
            # print ("(is_close(point_1, pos_1)", is_close(point_1, pos_1))
            # print ("(is_close(point_2, pos_2)", is_close(point_2, pos_2))
            # print("------------")
            

            # Check if the points match (considering floating point tolerance)
            if (is_close(point_1, pos_1) and is_close(point_2, pos_2)) or \
            (is_close(point_1, pos_2) and is_close(point_2, pos_1)):
                return edge

        return None

