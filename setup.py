from distutils.core import setup

setup(
    name='pynpoint_ifs',
    version='0.0.1',
    packages=['pynpoint_ifs'],
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
                'photutils',
                'skycalc_ipy'],
)

# Dependencies: Pynpoint, petitRADTRANS?, spectres?
