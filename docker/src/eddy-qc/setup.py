from setuptools import setup,find_packages
with open('requirements.txt', 'rt') as f:
    install_requires = [l.strip() for l in f.readlines()]

setup(name='eddy_qc',
	version='1.0.0',
	description='EDDY Quality Control tools',
	author='Matteo Bastiani',
	install_requires=install_requires,
    scripts=['eddy_qc/scripts/eddy_squad','eddy_qc/scripts/eddy_quad'],
	packages=find_packages())
