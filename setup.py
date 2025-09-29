from distutils.core import setup

setup(
    name='pynpoint-ifs',
    version='0.1.0',
    packages=['pynpoint-ifs'],
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
