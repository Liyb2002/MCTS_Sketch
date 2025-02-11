import json
import os
from copy import deepcopy

import Preprocessing.proc_CAD.build123.protocol
from Preprocessing.proc_CAD.basic_class import Face, Edge, Vertex
import Preprocessing.proc_CAD.helper


class parsed_program():
    def __init__(self, file_path, data_directory = None, output = True):
        self.file_path = file_path
        self.data_directory = data_directory

        if not data_directory:
            self.data_directory = os.path.dirname(__file__)

        canvas_directory = os.path.join(self.data_directory, 'canvas')
        os.makedirs(canvas_directory, exist_ok=True)

        self.canvas = None
        self.prev_sketch = None
        self.Op_idx = 0
        self.output = output
        
    def read_json_file(self):
        with open(self.file_path, 'r') as file:
            data = json.load(file)
            self.len_program = len(data)
            for i in range(len(data)):
                Op = data[i]
                operation = Op['operation']
                
                if operation[0] == 'sketch':
                    self.parse_sketch(Op)
                
                if operation[0] == 'extrude':
                    self.parse_extrude(Op, data[i-1])
                
                if operation[0] == 'fillet':
                    self.parse_fillet(Op)
                
                if operation[0] == 'chamfer':
                    self.parse_chamfer(Op)
 
                if operation[0] == 'terminate':
                    self.Op_idx += 1
                    break

        return
            

    def parse_sketch(self, Op):
        if 'radius' in Op['faces'][0]:
            self.parse_circle(Op)
            return 

        point_list = [vert['coordinates'] for vert in Op['vertices']]
        
        new_point_list = [point_list[0]]  # Start with the first point
        for i in range(1, len(point_list)):
            # Append each subsequent point twice
            new_point_list.append(point_list[i])
            new_point_list.append(point_list[i])
        
        # Add the first point again at the end to close the loop
        new_point_list.append(point_list[0])

        self.prev_sketch = Preprocessing.proc_CAD.build123.protocol.build_sketch(self.Op_idx, self.canvas, new_point_list, self.output, self.data_directory)
        self.Op_idx += 1

    def parse_circle(self, Op):
        radius = Op['faces'][0]['radius']
        center = Op['faces'][0]['center']
        normal = Op['faces'][0]['normal']

        if normal[0] == 0 and normal[1] == 0 and normal[2] == 0:
            normal[0] = 1

        self.prev_sketch = Preprocessing.proc_CAD.build123.protocol.build_circle(self.Op_idx, radius, center, normal, self.output, self.data_directory)
        self.Op_idx += 1
        self.circle_center = center
        
    def parse_extrude(self, Op, sketch_Op):

        sketch_point_list = [vert['coordinates'] for vert in sketch_Op['vertices']]
        sketch_face_normal = sketch_Op['faces'][0]['normal']
        extrude_amount = Op['operation'][2]
        isSubtract = (extrude_amount < 0)
        
        
        # If it is circle
        if len(sketch_point_list) ==0:
            expected_axis, expected_value = Preprocessing.proc_CAD.helper.expected_lvl(self.circle_center, sketch_face_normal, extrude_amount)
            if not isSubtract: 
                canvas_1 = Preprocessing.proc_CAD.build123.protocol.test_extrude(self.prev_sketch, extrude_amount)
                canvas_2 = Preprocessing.proc_CAD.build123.protocol.test_extrude(self.prev_sketch, -extrude_amount)

                if (canvas_1 is not None) and Preprocessing.proc_CAD.helper.canvas_has_lvl(canvas_1, expected_axis, expected_value):
                    self.canvas = Preprocessing.proc_CAD.build123.protocol.build_extrude(self.Op_idx, self.canvas, self.prev_sketch, extrude_amount, self.output, self.data_directory)
                if (canvas_2 is not None) and Preprocessing.proc_CAD.helper.canvas_has_lvl(canvas_2, expected_axis, expected_value):
                    self.canvas = Preprocessing.proc_CAD.build123.protocol.build_extrude(self.Op_idx, self.canvas, self.prev_sketch, -extrude_amount, self.output, self.data_directory)

            else:
                self.canvas = Preprocessing.proc_CAD.build123.protocol.build_subtract(self.Op_idx, self.canvas, self.prev_sketch, extrude_amount, self.output, self.data_directory)



        # Not circle
        else:
            expected_point = Op['vertices'][0]['coordinates']        

            if not isSubtract: 
                canvas_1 = Preprocessing.proc_CAD.build123.protocol.test_extrude(self.prev_sketch, extrude_amount)
                canvas_2 = Preprocessing.proc_CAD.build123.protocol.test_extrude(self.prev_sketch, -extrude_amount)

                if (canvas_1 is not None) and Preprocessing.proc_CAD.helper.canvas_has_point(canvas_1, expected_point):
                    self.canvas = Preprocessing.proc_CAD.build123.protocol.build_extrude(self.Op_idx, self.canvas, self.prev_sketch, extrude_amount, self.output, self.data_directory)
                if (canvas_2 is not None) and Preprocessing.proc_CAD.helper.canvas_has_point(canvas_2, expected_point):
                    self.canvas = Preprocessing.proc_CAD.build123.protocol.build_extrude(self.Op_idx, self.canvas, self.prev_sketch, -extrude_amount, self.output, self.data_directory)

            else:
                self.canvas = Preprocessing.proc_CAD.build123.protocol.build_subtract(self.Op_idx, self.canvas, self.prev_sketch, extrude_amount, self.output, self.data_directory)

        self.Op_idx += 1
        
    def parse_fillet(self, Op):
        print("Do fillet!")
        fillet_amount = Op['operation'][2]['amount']
        verts = Op['operation'][3]['old_verts_pos']

        target_edge = Preprocessing.proc_CAD.helper.find_target_verts(verts, self.canvas.edges())

        if target_edge != None:
            self.canvas = Preprocessing.proc_CAD.build123.protocol.build_fillet(self.Op_idx, self.canvas, target_edge, fillet_amount, self.output, self.data_directory)
            self.Op_idx += 1
            
    def parse_chamfer(self, Op):
        chamfer_amount = Op['operation'][2]['amount']
        verts = Op['operation'][3]['old_verts_pos']

        target_edge = Preprocessing.proc_CAD.helper.find_target_verts(verts, self.canvas.edges())

        if target_edge != None:
            self.canvas = Preprocessing.proc_CAD.build123.protocol.build_chamfer(self.Op_idx, self.canvas, target_edge, chamfer_amount, self.output, self.data_directory)
            self.Op_idx += 1

    def is_valid_parse(self):
        return self.Op_idx == self.len_program 


# Example usage:

def run(data_directory = None):
    file_path = os.path.join(os.path.dirname(__file__), 'canvas', 'Program.json')
    if data_directory:
        file_path = os.path.join(data_directory, 'Program.json')

    parsed_program_class = parsed_program(file_path, data_directory)
    parsed_program_class.read_json_file()
    
    return parsed_program_class.is_valid_parse()
