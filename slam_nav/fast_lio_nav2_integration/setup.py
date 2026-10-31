import os
from glob import glob
from setuptools import setup

package_name = 'fast_lio_nav2_integration'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install all launch files
        (os.path.join('share', package_name, 'launch'),
         glob(os.path.join('launch', '*.launch.py'))),
        # Install all config files
        (os.path.join('share', package_name, 'config'),
         glob(os.path.join('config', '*.yaml')) +
         glob(os.path.join('config', '*.pgm')) +
         glob(os.path.join('config', '*.rviz'))),
        # cmd_vel -> XLeRobot base bridges (run with the lerobot env's Python)
        (os.path.join('share', package_name, 'scripts'),
         glob(os.path.join('scripts', '*.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jensen Lavering',
    maintainer_email='jela4948@colorado.edu',
    description='RTAB-Map SLAM and Nav2 navigation for the XLeRobot base with a RealSense depth camera.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'odom_quat_normalizer = fast_lio_nav2_integration.odom_quat_normalizer:main',
        ],
    },
)
