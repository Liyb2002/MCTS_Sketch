import numpy as np
import Preprocessing.proc_CAD.generate_program
import random



def random_program(data_directory = None):
    canvas_class = Preprocessing.proc_CAD.generate_program.Brep()

    #init a program
    canvas_class.init_sketch_op()
    canvas_class.extrude_op()

    if random.random() < 0.5:
        canvas_class.random_chamfer()
    else:
        canvas_class.random_fillet()
    


    #random gen for n steps
    steps = random.randint(3, 3)
    for _ in range(steps - 1):
        canvas_class.regular_sketch_op()
        canvas_class.extrude_op()

        fillet_times = random.randint(1, 1)
        for _ in range(fillet_times):
            if random.random() < 0.5:
                canvas_class.random_chamfer()
            else:
                canvas_class.random_fillet()


    canvas_class.write_to_json(data_directory)

def simple_gen(data_directory = None):
    canvas_class = Preprocessing.proc_CAD.generate_program.Brep()
    canvas_class.init_sketch_op()
    canvas_class.extrude_op()
    canvas_class.regular_sketch_op()
    canvas_class.extrude_op()

    canvas_class.write_to_json(data_directory)
