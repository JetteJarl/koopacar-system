from setuptools import setup

package_name = 'runtime_monitoring'

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
            'topic_monitoring_node = runtime_monitoring.topic_monitoring_node:main'
            'central_monitoring_node = runtime_monitoring.central_monitoring_node:main'
        ],
    },
)
