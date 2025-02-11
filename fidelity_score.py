import numpy as np
from scipy.spatial import cKDTree
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface

def read_step(filepath):
    """Reads a STEP file and returns the shape."""
    try:
        step_reader = STEPControl_Reader()
        step_reader.ReadFile(filepath)
        step_reader.TransferRoot(1)
        shape = step_reader.Shape()
        return shape
    except Exception as e:
        print(f"Error reading STEP file: {filepath}, {e}")
        return None

def sample_points_from_shape(shape, tolerance=0.01, sample_density=100):
    """
    Samples points from the surface of the shape using a given tolerance.
    """
    if shape is None:
        print("Shape is None, skipping sampling.")
        return []
    try:
        # Generate a mesh
        BRepMesh_IncrementalMesh(shape, tolerance)

        # Explore the faces and sample points
        points = []
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = topods.Face(explorer.Current())  # Static method for Face
            adaptor = BRepAdaptor_Surface(face)
            
            # Get the bounds of the surface
            umin, umax = adaptor.FirstUParameter(), adaptor.LastUParameter()
            vmin, vmax = adaptor.FirstVParameter(), adaptor.LastVParameter()

            # Sample points within the surface bounds
            u_step = (umax - umin) / sample_density
            v_step = (vmax - vmin) / sample_density

            u = umin
            while u <= umax:
                v = vmin
                while v <= vmax:
                    point = adaptor.Value(u, v)
                    points.append((point.X(), point.Y(), point.Z()))
                    v += v_step
                u += u_step
            explorer.Next()
        return points
    except Exception as e:
        print(f"Error sampling points from shape: {e}")
        return []

def chamfer_distance(points1, points2):
    """Computes the Chamfer distance between two sets of points."""
    if not points1 or not points2:
        print("Empty point cloud detected!")
        return float('inf')  # Return a high distance for empty point clouds
    points1 = np.array(points1)
    points2 = np.array(points2)
    try:
        # Create k-d trees for fast nearest neighbor search
        tree1 = cKDTree(points1)
        tree2 = cKDTree(points2)

        # Compute distances from points1 to points2
        dist1, _ = tree1.query(points2)

        # Compute distances from points2 to points1
        dist2, _ = tree2.query(points1)

        # Calculate Chamfer distance
        chamfer = np.mean(dist1) + np.mean(dist2)
        return chamfer
    except Exception as e:
        print(f"Error computing Chamfer distance: {e}")
        return float('inf')

def compute_fidelity_score(gt_brep_path, output_brep_path, tolerance=0.01, sample_density=20):
    """
    Computes the fidelity score based on Chamfer distances between two BREP files.
    
    Parameters:
        gt_brep_path (str): Path to the ground truth BREP file.
        output_brep_path (str): Path to the output BREP file.
        tolerance (float): Tolerance for sampling points.
        sample_density (int): Sampling density for generating points.

    Returns:
        float: Fidelity score based on the Chamfer distances.
    """
    try:

        # Read shapes from STEP files
        gt_shape = read_step(gt_brep_path)
        output_shape = read_step(output_brep_path)

        if gt_shape is None or output_shape is None:
            print("Invalid shape detected, skipping fidelity computation.")
            return 0

        # Sample points from both shapes
        gt_points = sample_points_from_shape(gt_shape, tolerance, sample_density)
        output_points = sample_points_from_shape(output_shape, tolerance, sample_density)

        if not gt_points or not output_points:
            print("Insufficient points sampled, skipping fidelity computation.")
            return 0

        # Compute Chamfer distances
        gt_to_output = chamfer_distance(gt_points, output_points)
        output_to_gt = chamfer_distance(output_points, gt_points)

        # Calculate fidelity score
        fidelity_score = 1 / (1 + gt_to_output + output_to_gt)

        return fidelity_score
    except Exception as e:
        print(f"Error computing fidelity score: {e}")
        return 0
