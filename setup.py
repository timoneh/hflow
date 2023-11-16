from setuptools import setup

setup(
    name="hflow",
    version="1.0",
    author="Heikki Timonen",
    packages = ['hflow', 'hflow.metrics'],
    package_dir = {
        'hflow': 'hflow',
        'hflow.metrics': 'hflow/metrics'
    }
)