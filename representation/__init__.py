from .base_class import CGraph, TGraph, Representation
from .slg import StripsLearningGraph
from .dlg import DeleteLearningGraph
from .flg import FdrLearningGraph
from .llg import LiftedLearningGraph
from .lwg import LiftedWalkGraph
from .ltg import LiftedTopologyGraph
from .clg import ContrastiveLearningGraph


REPRESENTATIONS = {
  "slg": StripsLearningGraph,
  "dlg": DeleteLearningGraph,
  "flg": FdrLearningGraph,
  "llg": LiftedLearningGraph,
  "lwg": LiftedWalkGraph,
  "ltg": LiftedTopologyGraph,
  "clg": ContrastiveLearningGraph
}

REPRESENTATIONS_STR = ["dlg", "slg", "flg", "llg", "lwg", "ltg", "clg"]

