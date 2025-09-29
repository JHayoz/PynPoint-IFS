# PynPoint-IFS
Python package to incorporate High-Contrast Integral Field Spectroscopic data into PynPoint. This package enables the following reduction and analysis steps:
- Reading of .FITS files from IFS data
- Selection of wavelength
- Bad pixel correction
- Frame cropping
- Frame selection
- Frame alignment and derotation
- PSF subtraction using spectral PCA or PCA/ADI
- Computation of molecular maps
- Computation of detection significance
- Computation of detection limits

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Installation

pip install pynpoint-ifs

or clone from source:

git clone git@github.com:JHayoz/PynPoint-IFS.git
cd PynPoint-IFS
pip install .

## Usage

Follow the instructions detailed in the Jupyter notebook `example.ipynb` for a step-by-step guide.

## Contributing
Contributions are welcome! Please open issues or submit pull requests.

## License
This project is licensed under the GNU License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- If you make use of this package, please cite Hayoz et al. 2025, submitted.
- Uses `numpy`, `astropy`, `scipy`, `PynPoint`.
