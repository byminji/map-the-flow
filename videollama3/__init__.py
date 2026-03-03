import os

# The inner videollama3 subpackage uses absolute imports like `from videollama3.constants import ...`
# expecting itself to be the top-level package. Adding its directory to __path__ lets Python
# find those submodules correctly while keeping `videollama3.videollama3` accessible.
__path__.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "videollama3"))