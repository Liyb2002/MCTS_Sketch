
import Preprocessing.proc_CAD.proc_gen
import Preprocessing.proc_CAD.CAD_to_stroke_cloud
import Preprocessing.proc_CAD.render_images
import Preprocessing.proc_CAD.helper
import Preprocessing.proc_CAD.render_images
import Preprocessing.proc_CAD.draw_all_lines_baseline

import Preprocessing.gnn_graph
import Preprocessing.SBGCN.brep_read

import Preprocessing.proc_CAD.cad2sketch_stroke_features

import Preprocessing.generate_dataset_baseline

from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCP.StlAPI import StlAPI_Reader
from OCP.TopoDS import TopoDS_Shape
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.IFSelect import IFSelect_RetDone

import json
import shutil
import os
import pickle
import torch
import numpy as np
import threading
import re
import trimesh



class cad2sketch_dataset_generator():

    def __init__(self):
        self.data_path = os.path.join(os.getcwd(), 'dataset', 'cad2sketch')
        self.target_path = os.path.join(os.getcwd(), 'dataset', 'cad2sketch_annotated')
        self.idx = 0


        self.generate_dataset()


    def generate_dataset(self):
        folders = [folder for folder in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, folder))]
        

        for folder in folders:
            # folder_path = 'dataset/cad2sketch/201'
            folder_path = os.path.join(self.data_path, folder)

            subfolders = [sf for sf in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, sf))]
            if not subfolders:
                print(f"  No subfolders found in '{folder}'. Skipping...")
                continue
            
            for subfolder in subfolders:
                # subfolder_path = 'dataset/cad2sketch/201/51.6_-136.85_1.4'
                subfolder_path = os.path.join(folder_path, subfolder)
                
                self.process_subfolder(folder_path, subfolder_path)
    
    
    def process_subfolder(self, folder_path, subfolder_path):
        json_file_path = os.path.join(subfolder_path, 'final_edges.json')
        all_edges_file_path = os.path.join(subfolder_path, 'all_edges.json')
        strokes_dict_path = os.path.join(subfolder_path, 'strokes_dict.json')

        if not os.path.exists(json_file_path):
            return        
    
        # Create a new folder 'data_{idx}' in the target path
        new_folder_name = f"data_{self.idx}"
        new_folder_path = os.path.join(self.target_path, new_folder_name)
        os.makedirs(new_folder_path, exist_ok=True)
        
        
        # Create '/canvas' and '/shape_info' subdirectories
        target_path = os.path.join(new_folder_path, 'canvas')
        shape_info_folder = os.path.join(new_folder_path, 'shape_info')
        os.makedirs(target_path, exist_ok=True)
        os.makedirs(shape_info_folder, exist_ok=True)

        self.copy_shape_files(folder_path, target_path)
        self.idx += 1

        # Node connection_matrix
        strokes_dict_data = self.read_json(strokes_dict_path)
        connected_stroke_nodes = self.compute_connection_matrix(strokes_dict_data)

        # Node Features
        json_data = self.read_json(json_file_path)
        Preprocessing.proc_CAD.cad2sketch_stroke_features.vis_final_edges(json_data)

        all_edges_data = self.read_json(all_edges_file_path)
        Preprocessing.proc_CAD.cad2sketch_stroke_features.via_all_edges(all_edges_data)

        # self.compute_shape_info(json_data, connected_stroke_nodes, target_path, shape_info_folder)



    def compute_shape_info(self, json_data, connected_stroke_nodes, target_path, shape_info_folder):

        # 1) Produce the Stroke Cloud features            
        stroke_node_features = Preprocessing.proc_CAD.cad2sketch_stroke_features.build_final_edges_json(json_data)
        stroke_operations_order_matrix = self.compute_opertations_order_matrix(json_data)

        strokes_perpendicular, strokes_non_perpendicular =  Preprocessing.proc_CAD.helper.stroke_relations(stroke_node_features, connected_stroke_nodes)


        # 2) Get the loops
        stroke_cloud_loops = Preprocessing.proc_CAD.helper.face_aggregate_networkx(stroke_node_features) + Preprocessing.proc_CAD.helper.face_aggregate_circle(stroke_node_features)
        stroke_cloud_loops = [list(loop) for loop in stroke_cloud_loops]

        # 3) Compute Loop Neighboring Information
        loop_neighboring_all = Preprocessing.proc_CAD.helper.loop_neighboring_simple(stroke_cloud_loops)
        loop_neighboring_vertical = Preprocessing.proc_CAD.helper.loop_neighboring_complex(stroke_cloud_loops, stroke_node_features, loop_neighboring_all)
        loop_neighboring_horizontal = Preprocessing.proc_CAD.helper.coplanr_neighorbing_loop(loop_neighboring_all, loop_neighboring_vertical)
        loop_neighboring_contained = Preprocessing.proc_CAD.helper.loop_contained(stroke_cloud_loops, stroke_node_features)


        # 4) Load Brep
        brep_files = [f for f in os.listdir(target_path) if f.lower().endswith('.step')]
        brep_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

        final_brep_edges = []
        final_cylinder_features = []
        new_features = []
        file_count = 0
        for file_name in brep_files:
            print("working on", file_name)
            brep_file_path = os.path.join(target_path, file_name)
            edge_features_list, cylinder_features = Preprocessing.SBGCN.brep_read.create_graph_from_step_file(brep_file_path)
            print("edge_features_list", len(edge_features_list))

            # If this is the first brep
            if len(final_brep_edges) == 0:
                final_brep_edges = edge_features_list
                final_cylinder_features = cylinder_features
            else:
                # We already have brep
                new_features= Preprocessing.generate_dataset_baseline.find_new_features(final_brep_edges, edge_features_list) 
                final_brep_edges += new_features
                final_cylinder_features += cylinder_features
            
            # Preprocessing.proc_CAD.helper.vis_brep(np.array(edge_features_list))
            output_brep_edges = Preprocessing.proc_CAD.helper.pad_brep_features(final_brep_edges + final_cylinder_features)
            brep_loops = Preprocessing.proc_CAD.helper.face_aggregate_networkx(output_brep_edges) + Preprocessing.proc_CAD.helper.face_aggregate_circle_brep(output_brep_edges)
            # brep_loops = [list(loop) for loop in brep_loops]
        
            # # 5) Stroke_Cloud - Brep Connection
            # stroke_to_loop_lines = Preprocessing.proc_CAD.helper.stroke_to_brep(stroke_cloud_loops, brep_loops, stroke_node_features, output_brep_edges)
            # stroke_to_loop_circle = Preprocessing.proc_CAD.helper.stroke_to_brep_circle(stroke_cloud_loops, brep_loops, stroke_node_features, output_brep_edges)
            # stroke_to_loop = Preprocessing.proc_CAD.helper.union_matrices(stroke_to_loop_lines, stroke_to_loop_circle)
            
            # stroke_to_edge_lines = Preprocessing.proc_CAD.helper.stroke_to_edge(stroke_node_features, output_brep_edges)
            # stroke_to_edge_circle = Preprocessing.proc_CAD.helper.stroke_to_edge_circle(stroke_node_features, output_brep_edges)
            # stroke_to_edge = Preprocessing.proc_CAD.helper.union_matrices(stroke_to_edge_lines, stroke_to_edge_circle)

            # # 7) Write the data to file
            # output_file_path = os.path.join(shape_info_folder, f'shape_info_{file_count}.pkl')
            # with open(output_file_path, 'wb') as f:
            #     pickle.dump({
            #         'stroke_cloud_loops': stroke_cloud_loops, 

            #         'stroke_node_features': stroke_node_features,
            #         'stroke_type_features': None,
            #         'strokes_perpendicular': strokes_perpendicular,
            #         'output_brep_edges': output_brep_edges,
            #         'stroke_operations_order_matrix': stroke_operations_order_matrix, 

            #         'loop_neighboring_vertical': loop_neighboring_vertical,
            #         'loop_neighboring_horizontal': loop_neighboring_horizontal,
            #         'loop_neighboring_contained': loop_neighboring_contained,

            #         'stroke_to_loop': stroke_to_loop,
            #         'stroke_to_edge': stroke_to_edge
            #     }, f)
            
            # file_count += 1

        print("DONE")


    def compute_connection_matrix(self, json_data):
        # Extract all unique IDs
        ids = [d['id'] for d in json_data]
        id_to_index = {id_: index for index, id_ in enumerate(ids)}

        # Initialize the matrix with zeros
        matrix_size = len(ids)
        connection_matrix = np.zeros((matrix_size, matrix_size), dtype=int)

        # Populate the connection matrix
        for dict_item in json_data:
            id_ = dict_item['id']
            intersections = dict_item['intersections']
            id_index = id_to_index[id_]
            
            # If intersections is a list of lists, flatten it
            for sublist in intersections:
                if isinstance(sublist, list):
                    for sub_id in sublist:
                        if sub_id in id_to_index:
                            sublist_index = id_to_index[sub_id]
                            connection_matrix[id_index][sublist_index] = 1
                            connection_matrix[sublist_index][id_index] = 1  # For undirected connection
                elif sublist in id_to_index:
                    sublist_index = id_to_index[sublist]
                    connection_matrix[id_index][sublist_index] = 1
                    connection_matrix[sublist_index][id_index] = 1  # For undirected connection

        return connection_matrix


    def compute_opertations_order_matrix(self, json_data):
        all_feature_ids = set()
        
        for stroke in json_data.values():
            feature_id = stroke['feature_id']
            if isinstance(feature_id, list):
                all_feature_ids.update(feature_id)
            else:
                all_feature_ids.add(feature_id)
        
        feature_list = sorted(all_feature_ids) 
        feature_index = {feature: idx for idx, feature in enumerate(feature_list)}
        
        num_strokes = len(json_data)
        num_features = len(feature_list)
        matrix = np.zeros((num_strokes, num_features), dtype=int)
        
        # Populate the matrix 
        for stroke_idx, stroke in enumerate(json_data.values()):
            feature_id = stroke['feature_id']
            if isinstance(feature_id, list):
                for feature in feature_id:
                    matrix[stroke_idx][feature_index[feature]] = 1
            else:
                matrix[stroke_idx][feature_index[feature_id]] = 1
        
        return matrix



    def copy_shape_files(self, source_path, target_path):
        shape_files = [f for f in os.listdir(source_path) if f.lower().endswith('.stl')]
                
        for stl_file in shape_files:
            # Copy the .stl file
            source_file = os.path.join(source_path, stl_file)
            target_stl_file = os.path.join(target_path, stl_file)
            shutil.copy(source_file, target_stl_file)

            step_file_name = os.path.splitext(stl_file)[0] + ".step"
            target_step_file = os.path.join(target_path, step_file_name)

            if not self.convert_stl_to_step(target_stl_file, target_step_file):
                print(f"Failed to convert {stl_file} to STEP format.")
            else:
                print(f"Successfully converted {stl_file} to {step_file_name}.")


    def convert_stl_to_step(self, stl_file, step_file):
        """
        Converts an .stl file to .step using Open CASCADE.
        """
        # Read the STL file
        stl_reader = StlAPI_Reader()
        shape = TopoDS_Shape()
        if not stl_reader.Read(shape, stl_file):
            print(f"Error reading STL file: {stl_file}")
            return False
        
        # Perform meshing (optional but recommended)
        BRepMesh_IncrementalMesh(shape, 0.1)

        # Write to STEP file
        step_writer = STEPControl_Writer()
        step_writer.Transfer(shape, STEPControl_AsIs)
        status = step_writer.Write(step_file)
        
        return status == IFSelect_RetDone
    
    

    def read_json(self, file_path):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
            return data
        except Exception as e:
            print(f"Error reading JSON file {file_path}: {e}")
            return None


