from . import builtin  # ensure the builtin datasets are registered
from .dataset_mapper import DatasetMapperWithBasis


__all__ = ["DatasetMapperWithBasis"]
#grid mask trick
#https://github.com/Jia-Research-Lab/GridMask/blob/master/detection_grid/maskrcnn_benchmark/data/transforms/grid.py