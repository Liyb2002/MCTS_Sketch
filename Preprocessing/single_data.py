import proc_CAD.proc_gen
import proc_CAD.CAD_to_stroke_cloud
import proc_CAD.render_images
import proc_CAD.Program_to_STL
import proc_CAD.helper

import gnn_graph
import SBGCN.run_SBGCN

import os
import pickle

class single_data:
    def __init__(self, stroke_cloud_graph=None, brep_embedding=None, operations=None):

        #generate a new program
        # proc_CAD.proc_gen.random_program()
        proc_CAD.proc_gen.simple_gen()

        #create brep for the new program
        proc_CAD.Program_to_STL.run()


        # self.stroke_cloud= proc_CAD.CAD_to_stroke_cloud.run(vis = True)
        # self.stroke_cloud_graph = gnn_graph.build_graph(self.stroke_cloud)
        # stroke_cloud_save_path = os.path.join(os.getcwd(), 'proc_CAD', 'canvas', 'sketch_hetero_data.pkl')
        # with open(stroke_cloud_save_path, 'wb') as f:
        #     pickle.dump(self.stroke_cloud_graph, f)


        # program_file_path = os.path.join(os.getcwd(), 'proc_CAD', 'canvas', 'program.json')

        # self.program = proc_CAD.helper.program_to_string(program_file_path)

        # self.SBGCN_encoder = SBGCN.run_SBGCN.load_pretrained_SBGCN_model()

        # self.brep_embedding(0)

        # proc_CAD.render_images.run_render_images()


    def brep_embedding(self, idx):
        brep_file_path = os.path.join(os.getcwd(), 'proc_CAD', 'canvas', f'brep_{idx}.step')
        brep_graph = SBGCN.brep_read.create_graph_from_step_file(brep_file_path)
        embedding = self.SBGCN_encoder.embed(brep_graph)

        return embedding








single_data_example = single_data()

