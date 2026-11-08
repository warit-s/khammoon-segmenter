"""khammoon — Thai morpheme (คำมูล) segmentation with a small CPU model."""
from .segmenter import Segmenter, segment

__version__ = "0.1.0"
__all__ = ["Segmenter", "segment", "__version__"]
