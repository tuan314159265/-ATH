"""
Modeling Package for Customer Analytics

This package contains modules for RFM analysis, clustering, supervised learning,
and visualization of customer segmentation data.
"""

from .rfm_analysis import *
from .clustering import *
from .supervised_models import *
from .visualization import *

__version__ = "1.0.0"
__all__ = [
    # RFM Analysis
    'load_data', 'calculate_rfm', 'calculate_rfm_vanglai', 'remove_outliers_rfm',
    'preprocess_rfm_only', 'find_best_k', 'extract_vanglai',

    # Clustering
    'clustering_and_split_id_only', 'perform_clustering',

    # Supervised Models
    'load_data', 'prepare_features', 'evaluate', 'train_decision_tree',
    'train_random_forest', 'save_models',

    # Visualization
    'plot_elbow_method', 'plot_3d_clusters', 'plot_boxcox_transformation',
    'plot_segment_analysis', 'compare_models'
]