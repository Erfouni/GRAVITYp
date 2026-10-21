"""
GRAVITYp: Python package to study the optical appearance of spherically symmetric
compact objects in General Relativity.

Source:    https://github.com/alvaroslzar/GRAVITYp
"""

from .compact_object import *
from .zoo import *
from .geodesic_integration import *
from .ray_tracing import *
from .rings import *
from .transfer_functions import *
from .emission_models import *
from .shadows import *
from .plots import *

from . import (compact_object, zoo, geodesic_integration, ray_tracing, rings,
               transfer_functions, emission_models, shadows, plots)

# Written as `+=` statements so static analysers (e.g. Pylance) can follow them
__all__ = []
__all__ += compact_object.__all__
__all__ += zoo.__all__
__all__ += geodesic_integration.__all__
__all__ += ray_tracing.__all__
__all__ += rings.__all__
__all__ += transfer_functions.__all__
__all__ += emission_models.__all__
__all__ += shadows.__all__
__all__ += plots.__all__

import email.utils
from importlib.metadata import metadata, PackageNotFoundError

__name__ = "gravityp"

try:
    _meta = metadata(__name__)
    __version__ = _meta.get("Version")

    # setuptools packs pyproject.toml authors into the 'Author-email' header
    _author_email_header = _meta.get("Author-email")

    if _author_email_header:
        # Parse header into [("Name", "email@domain.com"), ...]
        _parsed_contacts = email.utils.getaddresses([_author_email_header])

        _names = [name for name, _ in _parsed_contacts if name]
        _emails = [addr for _, addr in _parsed_contacts if addr]

        # Format as strings so help(gravityp) displays them cleanly
        __author__ = ", ".join(_names) if _names else _meta.get("Author")
        __email__ = ", ".join(_emails) if _emails else None
    else:
        __author__ = _meta.get("Author")
        __email__ = None

except PackageNotFoundError:
    __version__ = None
    __author__ = None
    __email__ = None
