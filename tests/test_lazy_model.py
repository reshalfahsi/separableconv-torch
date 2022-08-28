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


def test_lazy_model1d():
    input = torch.randn(10, 3, 224)

    error_message = ""

    try:
        model = nn.Sequential(
            nn.LazyConv1d(16, 3, padding=1, stride=2),
            nn.LazySeparableConv1d(24, 3, padding=1, stride=2, depth_multiplier=2),
            nn.LazySeparableConv1d(32, 3, padding=1, stride=2),
            nn.LazySeparableConv1d(40, 3, padding=1, stride=2, depth_multiplier=0.5),
            nn.LazySeparableConv1d(48, 3, padding=1, stride=2),
        )

        output = model(input)
    except Exception as e:
        error_message = str(e)

    assert error_message == "depth_multiplier must be integer>=1"

    model = nn.Sequential(
        nn.LazyConv1d(16, 3, padding=1, stride=2),
        nn.LazySeparableConv1d(24, 3, padding=1, stride=2, depth_multiplier=2),
        nn.LazySeparableConv1d(32, 3, padding=1, stride=2),
        nn.LazySeparableConv1d(40, 3, padding=1, stride=2, depth_multiplier=3),
        nn.LazySeparableConv1d(48, 3, padding=1, stride=2),
    )

    output = model(input)

    assert output.shape == torch.Size([10, 48, 7])


def test_lazy_model2d():
    input = torch.randn(10, 3, 224, 224)

    error_message = ""

    try:
        model = nn.Sequential(
            nn.LazyConv2d(16, 3, padding=1, stride=2),
            nn.LazySeparableConv2d(24, 3, padding=1, stride=2, depth_multiplier=2),
            nn.LazySeparableConv2d(32, 3, padding=1, stride=2),
            nn.LazySeparableConv2d(40, 3, padding=1, stride=2, depth_multiplier=0.5),
            nn.LazySeparableConv2d(48, 3, padding=1, stride=2),
        )

        output = model(input)
    except Exception as e:
        error_message = str(e)

    assert error_message == "depth_multiplier must be integer>=1"

    model = nn.Sequential(
        nn.LazyConv2d(16, 3, padding=1, stride=2),
        nn.LazySeparableConv2d(24, 3, padding=1, stride=2, depth_multiplier=2),
        nn.LazySeparableConv2d(32, 3, padding=1, stride=2),
        nn.LazySeparableConv2d(40, 3, padding=1, stride=2, depth_multiplier=3),
        nn.LazySeparableConv2d(48, 3, padding=1, stride=2),
    )

    output = model(input)

    assert output.shape == torch.Size([10, 48, 7, 7])


def test_lazy_model3d():
    input = torch.randn(10, 3, 224, 224, 224)

    error_message = ""

    try:
        model = nn.Sequential(
            nn.LazyConv3d(16, 3, padding=1, stride=2),
            nn.LazySeparableConv3d(24, 3, padding=1, stride=2, depth_multiplier=2),
            nn.LazySeparableConv3d(32, 3, padding=1, stride=2),
            nn.LazySeparableConv3d(40, 3, padding=1, stride=2, depth_multiplier=0.5),
            nn.LazySeparableConv3d(48, 3, padding=1, stride=2),
        )

        output = model(input)
    except Exception as e:
        error_message = str(e)

    assert error_message == "depth_multiplier must be integer>=1"

    model = nn.Sequential(
        nn.LazyConv3d(16, 3, padding=1, stride=2),
        nn.LazySeparableConv3d(24, 3, padding=1, stride=2, depth_multiplier=2),
        nn.LazySeparableConv3d(32, 3, padding=1, stride=2),
        nn.LazySeparableConv3d(40, 3, padding=1, stride=2, depth_multiplier=3),
        nn.LazySeparableConv3d(48, 3, padding=1, stride=2),
    )

    output = model(input)

    assert output.shape == torch.Size([10, 48, 7, 7, 7])


if __name__ == "__main__":
    pytest.main([__file__])
