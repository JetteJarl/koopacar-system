from setuptools import setup

package_name = 'localization'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='koopacar_team',
    maintainer_email='ubuntu@koopacar.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'localization = localization.localization_node:main',
            'mapping = localization.mapping_node:main',
            'sensor_fusion = localization.sensor_fusion_node:main'
        ],
    },
)
