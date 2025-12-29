import warnings

# Suppress specific deprecation warnings
warnings.filterwarnings(
    "ignore", 
    category=DeprecationWarning, 
    module="sksurv"
)
warnings.filterwarnings(
    "ignore", 
    category=DeprecationWarning, 
    module="lifelines"
)

# Optionally: Convert UserWarnings to log messages
import logging
logging.captureWarnings(True)