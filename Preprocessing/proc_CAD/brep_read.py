from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopoDS import TopoDS_Shape, topods
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.BRepTools import breptools
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse, GeomAbs_CurveType
from OCC.Core.GCPnts import GCPnts_UniformAbscissa
from OCC.Core.GeomAdaptor import GeomAdaptor_Curve

from torch.utils.data import Dataset
from itertools import combinations

import torch
import os

def read_step_file(filename):

    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(filename)
    
    if status == 1:  # Check if the read was successful
        step_reader.TransferRoot()  # Transfers the whole STEP file
        shape = step_reader.Shape()  # Retrieves the translated shape
        return shape
    else:
        raise Exception("Error reading STEP file.")


def create_edge_node(edge):
    properties = GProp_GProps()
    brepgprop.LinearProperties(edge, properties)

    vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)

    verts = []
    while vertex_explorer.More():
        vertex = topods.Vertex(vertex_explorer.Current())
        pt = BRep_Tool.Pnt(vertex)
        verts.append([pt.X(), pt.Y(), pt.Z()])
        vertex_explorer.Next()

    edge_features = [verts[0][0], verts[0][1], verts[0][2], verts[1][0], verts[1][1], verts[1][2]]


    curve_adaptor = BRepAdaptor_Curve(edge)
    curve_type = curve_adaptor.GetType()

    is_curve = curve_type != GeomAbs_Line

    edge_features = {
        'vertices': verts,
        'type': 'feature_line',
        'is_curve': is_curve,
        'sampled_points': [],
        'projected_edge': [],
        'sigma' : 0.0,
        'mu': 0.0
    }

    if is_curve:
        num_points = 10

        sampler = GCPnts_UniformAbscissa(curve_adaptor, num_points)
        if sampler.IsDone():
            for i in range(1, num_points + 1):
                p = curve_adaptor.Value(sampler.Parameter(i))
                edge_features['sampled_points'].append([p.X(), p.Y(), p.Z()])

    return edge_features


def check_duplicate(new_feature, feature_list, face = 0):
    for existing_feature in feature_list:
        if existing_feature == new_feature:
            return 0
    
    return -1


def create_graph_from_step_file(step_path):
    shape = read_step_file(step_path)

    edge_features_list = []
    
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        face = topods.Face(face_explorer.Current())

        # Explore edges of the face
        edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
        while edge_explorer.More():
            edge = topods.Edge(edge_explorer.Current())
            edge_features = create_edge_node(edge)

            edge_duplicate_id = check_duplicate(edge_features, edge_features_list)
            if edge_duplicate_id != -1:
                edge_explorer.Next()
                continue
            
            edge_features_list.append(edge_features)
            
            edge_explorer.Next()
        
        
        face_explorer.Next()
    
    return edge_features_list

