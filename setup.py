# MIT License
#
# Copyright (c) 2022 Resha Dwika Hefni Al-Fahsi
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================


import sys
import platform
from pathlib import Path
from setuptools import setup, find_packages


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


# read version
version = {}
version_file_contents = (this_directory / "separableconv" / "version.py").read_text()
exec(version_file_contents, version)


# From: https://github.com/pytorch/pytorch/blob/master/setup.py


python_min_version = (3, 7, 0)
python_min_version_str = ".".join(map(str, python_min_version))
if sys.version_info < python_min_version:
    print(
        "You are using Python {}. Python >={} is required.".format(
            platform.python_version(), python_min_version_str
        )
    )
    sys.exit(-1)


version_range_max = max(sys.version_info[1], 9) + 1


setup(
    name="separableconv-torch",
    version=version["__version__"],
    author="Resha Dwika Hefni Al-Fahsi",
    author_email="resha.alfahsi@gmail.com",
    description="PyTorch implementation of Depthwise Separable Convolution",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/reshalfahsi/separableconv-torch",
    packages=find_packages(exclude=["tests"]),
    python_requires=">={}".format(python_min_version_str),
    install_requires=[
        "numpy",
        "torch",
        "torchvision",
    ],
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
    ]
    + [
        "Programming Language :: Python :: 3.{}".format(i)
        for i in range(python_min_version[1], version_range_max)
    ],
    keywords="pytorch, machine learning",
)
