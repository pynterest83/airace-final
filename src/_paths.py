"""Import bootstrap for the contest tree.

Every module under src/ starts with:

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa

which puts src/ on the path and then lets this file register the four
package dirs, so `from dataset import SceneData` or `import fuse_test`
works from anywhere without relative-depth games.
"""
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)

for _d in ("gs", "refine", "geom", "post"):
    _p = os.path.join(SRC, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
