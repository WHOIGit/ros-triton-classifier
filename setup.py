## ! DO NOT MANUALLY INVOKE THIS FILE, USE CATKIN INSTEAD

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    # triton_api is a plain module next to the node script, not a package.
    py_modules=['triton_api'],
    package_dir={'': 'src'},
)

setup(**setup_args)
