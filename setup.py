from distutils.core import setup

setup(
    name='Pynpoint_ifs',
    version='0.1',
    packages=['Pynpoint_ifs'],
    url='',
    license='',
    author='Emily O. Garvin',
    author_email='egarvin@phys.ethz.ch',
    description='Package for IFS cube reduction',
    install_requires=[
                'numpy',
                'scipy',
                'matplotlib',
                'astropy',
                'PyAstronomy',
                'pandas',
                'sklearn',
                'seaborn',
                'spectres',
                'petitRADTRANS',
                'photutils'],
)

# Dependencies: Pynpoint, petitRADTRANS?, spectres?