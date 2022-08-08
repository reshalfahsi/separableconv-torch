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


import pytest
import torch

import separableconv.nn as nn


def test_separableconv1d():
    input = torch.randn(4, 10, 100)
    m = nn.SeparableConv1d(10, 30, 3, padding=1)
    output = m(input)

    assert output.shape == torch.Size([4, 30, 100])


def test_separableconv2d():
    input = torch.randn(4, 10, 100, 100)
    m = nn.SeparableConv2d(10, 30, 3, padding=1)
    output = m(input)

    assert output.shape == torch.Size([4, 30, 100, 100])


def test_separableconv3d():
    input = torch.randn(4, 10, 100, 100, 100)
    m = nn.SeparableConv3d(10, 30, 3, padding=1)
    output = m(input)

    assert output.shape == torch.Size([4, 30, 100, 100, 100])


if __name__ == "__main__":
    pytest.main([__file__])
