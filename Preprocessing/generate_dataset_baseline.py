import Preprocessing.proc_CAD.proc_gen
import Preprocessing.proc_CAD.CAD_to_stroke_cloud
import Preprocessing.proc_CAD.render_images
import Preprocessing.proc_CAD.Program_to_STL
import Preprocessing.proc_CAD.helper
import Preprocessing.proc_CAD.render_images
import Preprocessing.proc_CAD.draw_all_lines_baseline

import Preprocessing.gnn_graph
import Preprocessing.SBGCN.brep_read

import shutil
import os
import pickle
import torch
import numpy as np
import threading
import re

class dataset_generator():

    def __init__(self):
        # if os.path.exists('dataset'):
        #     shutil.rmtree('dataset')

        self.dataset_name = 'dataset/test'
        os.makedirs(self.dataset_name, exist_ok=True)

        self.generate_dataset(self.dataset_name, number_data = 1, start =self.compute_start_idx())
    

    def compute_start_idx(self):
        data_path = os.path.join(os.getcwd(), self.dataset_name)
        data_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]

        pattern = re.compile(r'.*_(\d+)$')
        
        largest_number = 0
        
        # List all directories and retrieve the number at the end of the format
        for d in os.listdir(data_path):
            if os.path.isdir(os.path.join(data_path, d)):
                match = pattern.match(d)
                if match:
                    number = int(match.group(1))  # Extract the number
                    largest_number = max(largest_number, number)  # Keep track of the largest number

        
        return max(largest_number-1, 0)


    def generate_dataset(self, dir, number_data, start):
        successful_generations = start

        while successful_generations < number_data:

            if self.generate_single_data(successful_generations, dir):
                successful_generations += 1
            else:
                print("Retrying...")


    def generate_single_data(self, successful_generations, dir):
        data_directory = os.path.join(dir, f'data_{successful_generations}')
        if os.path.exists(data_directory):
            shutil.rmtree(data_directory)

        os.makedirs(data_directory, exist_ok=True)

        # Generate a new program & save the brep
        try:
            # Pass in the directory to the simple_gen function
            Preprocessing.proc_CAD.proc_gen.random_program(data_directory)
            # Preprocessing.proc_CAD.proc_gen.simple_gen(data_directory)

            # Create brep for the new program and pass in the directory
            valid_parse = Preprocessing.proc_CAD.Program_to_STL.run(data_directory)
            stroke_cloud_class = Preprocessing.proc_CAD.draw_all_lines_baseline.create_stroke_cloud_class(data_directory, False)
            stroke_cloud_class.read_all()

        except Exception as e:
            print(f"An error occurred: {e}")
            shutil.rmtree(data_directory)
            return False
        
        if not valid_parse:
            print("not valid valid_parse")
            shutil.rmtree(data_directory)
            return False
        
        
        print("----------------------")

        # 1) Produce the Stroke Cloud features            
        stroke_node_features, stroke_operations_order_matrix= Preprocessing.gnn_graph.build_graph(stroke_cloud_class.edges)

        stroke_type_features = Preprocessing.gnn_graph.build_stroke_type(stroke_cloud_class.edges)
        stroke_node_features, stroke_operations_order_matrix = Preprocessing.proc_CAD.helper.swap_rows_with_probability(stroke_node_features, stroke_operations_order_matrix)
        stroke_node_features = np.round(stroke_node_features, 4)
        
        connected_stroke_nodes = Preprocessing.proc_CAD.helper.connected_strokes(stroke_node_features)
        strokes_perpendicular, strokes_non_perpendicular =  Preprocessing.proc_CAD.helper.stroke_relations(stroke_node_features, connected_stroke_nodes)


        # 2) Get the loops
        stroke_cloud_loops = Preprocessing.proc_CAD.helper.face_aggregate_networkx(stroke_node_features) + Preprocessing.proc_CAD.helper.face_aggregate_circle(stroke_node_features)
        stroke_cloud_loops = Preprocessing.proc_CAD.helper.reorder_loops(stroke_cloud_loops)
        stroke_cloud_loops = [list(loop) for loop in stroke_cloud_loops]


        # 3) Compute Loop Neighboring Information
        loop_neighboring_all = Preprocessing.proc_CAD.helper.loop_neighboring_simple(stroke_cloud_loops)
        loop_neighboring_vertical = Preprocessing.proc_CAD.helper.loop_neighboring_complex(stroke_cloud_loops, stroke_node_features, loop_neighboring_all)
        loop_neighboring_horizontal = Preprocessing.proc_CAD.helper.coplanr_neighorbing_loop(loop_neighboring_all, loop_neighboring_vertical)
        loop_neighboring_contained = Preprocessing.proc_CAD.helper.loop_contained(stroke_cloud_loops, stroke_node_features)


        # 4) Load Brep
        brep_directory = os.path.join(data_directory, 'canvas')
        brep_files = [file_name for file_name in os.listdir(brep_directory)
            if file_name.startswith('brep_') and file_name.endswith('.step')]
        brep_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))


        final_brep_edges = []
        final_cylinder_features = []
        new_features = []
        file_count = 0
        for file_name in brep_files:
            brep_file_path = os.path.join(brep_directory, file_name)
            edge_features_list, cylinder_features = Preprocessing.SBGCN.brep_read.create_graph_from_step_file(brep_file_path)

            # If this is the first brep
            if len(final_brep_edges) == 0:
                final_brep_edges = edge_features_list
                final_cylinder_features = cylinder_features
            else:
                # We already have brep
                new_features= find_new_features(final_brep_edges, edge_features_list) 
                final_brep_edges += new_features
                final_cylinder_features += cylinder_features
            
            # Preprocessing.proc_CAD.helper.vis_brep(np.array(edge_features_list))
            output_brep_edges = Preprocessing.proc_CAD.helper.pad_brep_features(final_brep_edges + final_cylinder_features)
            brep_loops = Preprocessing.proc_CAD.helper.face_aggregate_networkx(output_brep_edges) + Preprocessing.proc_CAD.helper.face_aggregate_circle_brep(output_brep_edges)
            brep_loops = [list(loop) for loop in brep_loops]


            # 5) Stroke_Cloud - Brep Connection
            stroke_to_loop_lines = Preprocessing.proc_CAD.helper.stroke_to_brep(stroke_cloud_loops, brep_loops, stroke_node_features, output_brep_edges)
            stroke_to_loop_circle = Preprocessing.proc_CAD.helper.stroke_to_brep_circle(stroke_cloud_loops, brep_loops, stroke_node_features, output_brep_edges)
            stroke_to_loop = Preprocessing.proc_CAD.helper.union_matrices(stroke_to_loop_lines, stroke_to_loop_circle)
            
            stroke_to_edge_lines = Preprocessing.proc_CAD.helper.stroke_to_edge(stroke_node_features, output_brep_edges)
            stroke_to_edge_circle = Preprocessing.proc_CAD.helper.stroke_to_edge_circle(stroke_node_features, output_brep_edges)
            stroke_to_edge = Preprocessing.proc_CAD.helper.union_matrices(stroke_to_edge_lines, stroke_to_edge_circle)

            # 7) Write the data to file
            os.makedirs(os.path.join(data_directory, 'shape_info'), exist_ok=True)
            output_file_path = os.path.join(data_directory, 'shape_info', f'shape_info_{file_count}.pkl')
            with open(output_file_path, 'wb') as f:
                pickle.dump({
                    'stroke_cloud_loops': stroke_cloud_loops, 

                    'stroke_node_features': stroke_node_features,
                    'stroke_type_features': stroke_type_features,
                    'strokes_perpendicular': strokes_perpendicular,
                    'output_brep_edges': output_brep_edges,
                    'stroke_operations_order_matrix': stroke_operations_order_matrix, 

                    'loop_neighboring_vertical': loop_neighboring_vertical,
                    'loop_neighboring_horizontal': loop_neighboring_horizontal,
                    'loop_neighboring_contained': loop_neighboring_contained,

                    'stroke_to_loop': stroke_to_loop,
                    'stroke_to_edge': stroke_to_edge
                }, f)
            
            file_count += 1

        return True
        


def find_new_features(prev_brep_edges, new_edge_features):
    prev_brep_edges = [[round(coord, 4) for coord in line] for line in prev_brep_edges]
    new_edge_features = [[round(coord, 4) for coord in line] for line in new_edge_features]

    def is_same_direction(line1, line2):
        """Check if two lines have the same direction."""
        vector1 = np.array(line1[3:6]) - np.array(line1[:3])
        vector2 = np.array(line2[3:6]) - np.array(line2[:3])
        return np.allclose(vector1 / np.linalg.norm(vector1), vector2 / np.linalg.norm(vector2))

    def is_point_on_line(point, line):
        """Check if a point lies on a given line segment."""
        start, end = np.array(line[:3]), np.array(line[3:6])

        # Check if the point is collinear (still important to check)
        if not np.allclose(np.cross(end - start, point - start), 0):
            return False
        
        # Check if point lies within the bounds of the line segment
        min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
        min_y, max_y = min(start[1], end[1]), max(start[1], end[1])
        min_z, max_z = min(start[2], end[2]), max(start[2], end[2])
        
        return (min_x <= point[0] <= max_x) and (min_y <= point[1] <= max_y) and (min_z <= point[2] <= max_z)

    def is_line_contained(line1, line2):
        """Check if line1 is contained within line2."""
        return is_point_on_line(np.array(line1[:3]), line2) and is_point_on_line(np.array(line1[3:6]), line2)



    def find_unique_points(new_edge_line, prev_brep_line):
        """Find the two unique points between new_edge_line and prev_brep_line."""
        points = [
            tuple(new_edge_line[:3]),   # new_edge_line start
            tuple(new_edge_line[3:6]),   # new_edge_line end
            tuple(prev_brep_line[:3]),  # prev_brep_line start
            tuple(prev_brep_line[3:6]),  # prev_brep_line end
        ]

        # Find unique points
        unique_points = [point for point in points if points.count(point) == 1]

        # Ensure there are exactly two unique points
        if len(unique_points) == 2:
            return unique_points
        return None

    new_features = []

    for new_edge_line in new_edge_features:
        if new_edge_line[-1] != 0:
            new_features.append(new_edge_line)
            continue

        relation_found = False

        edge_start, edge_end = np.array(new_edge_line[:3]), np.array(new_edge_line[3:6])
        
        for prev_brep_line in prev_brep_edges:
            if prev_brep_line[-1] != 0:
                continue

            brep_start, brep_end = np.array(prev_brep_line[:3]), np.array(prev_brep_line[3:6])

            # This is a circle edge
            if (edge_start == edge_end).all():
                break

            # Check if the lines are the same, either directly or in reverse order
            if (np.allclose(edge_start, brep_start) and np.allclose(edge_end, brep_end)) or \
            (np.allclose(edge_start, brep_end) and np.allclose(edge_end, brep_start)):
                # Relation 1: The two lines are exactly the same
                relation_found = True
                break
            
            elif is_same_direction(new_edge_line, prev_brep_line) and is_line_contained(new_edge_line, prev_brep_line):
                # new feature is in prev brep
                relation_found = True
                
                unique_points = find_unique_points(new_edge_line, prev_brep_line)
                if unique_points:
                    # Create a new line using the unique points
                    new_line = list(unique_points[0]) + list(unique_points[1])
                    new_features.append(new_line)
                    relation_found = True
                    break
                break
            
            elif is_same_direction(new_edge_line, prev_brep_line) and is_line_contained(prev_brep_line, new_edge_line):
                # prev brep is in new feature
                relation_found = True
                
                unique_points = find_unique_points(new_edge_line, prev_brep_line)
                if unique_points:
                    # Create a new line using the unique points
                    new_line = list(unique_points[0]) + list(unique_points[1])
                    new_features.append(new_line)
                    relation_found = True
                    break

                break
        
        if not relation_found:
            # Relation 4: None of the relations apply
            new_features.append(new_edge_line)

    return new_features