from .dion import Dion
from .dion import DionMixedPrecisionConfig
from .dion_simple import Dion as DionSimple
from .dion_reference import Dion as DionReference
from .muon import Muon
from .muon_reference import Muon as MuonReference

# [GLUON PATCH]: Expose our new Gluon optimizer and utils at the top level
from .gluon import Gluon
from .gluon_utils import GluonProfiler, gluanalyze, save_gluon_config, load_gluon_config, gluon_that_model
from .glazy_gloptimizer import GlazyGloptimizer

print("✅ Loaded custom dion-gluon fork.") # Optional: a little message to confirm your fork is being used!