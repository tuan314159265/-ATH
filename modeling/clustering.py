import importlib.util
import os

_kmean_path = os.path.join(os.path.dirname(__file__), "k-mean.py")
if not os.path.exists(_kmean_path):
    raise ImportError(f"Missing k-mean module file: {_kmean_path}")

spec = importlib.util.spec_from_file_location("modeling._k_mean", _kmean_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load module spec for {_kmean_path}")

_kmean = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_kmean)

clustering_and_split_id_only = _kmean.clustering_and_split_id_only
perform_clustering = _kmean.perform_clustering
main = _kmean.main

__all__ = ["clustering_and_split_id_only", "perform_clustering", "main"]
