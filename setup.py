from distutils.core import setup

setup(
  name='orthosis_env_metapackage',
  version='1.0',
  description='Orthosis Controllers',
  author='Dhruv Kool Rajamani',
  author_email='dkoolrajamani@wpi.edu',
  packages=['controller'],
  install_requires=['gym'],
  package_dir={
    'controller': 'scripts',
    'gym_env': 'gym_env'
  }
)
