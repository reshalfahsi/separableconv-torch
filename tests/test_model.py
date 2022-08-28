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


def test_model1d():
    input = torch.randn(10, 3, 224)

    error_message = ""

    try:
        model = nn.Sequential(
            nn.Conv1d(3, 16, 3, padding=1, stride=2),
            nn.SeparableConv1d(16, 24, 3, padding=1, stride=2, depth_multiplier=2),
            nn.SeparableConv1d(24, 32, 3, padding=1, stride=2),
            nn.SeparableConv1d(32, 40, 3, padding=1, stride=2, depth_multiplier=0.5),
            nn.SeparableConv1d(40, 48, 3, padding=1, stride=2),
        )
    except Exception as e:
        error_message = str(e)

    assert error_message == "depth_multiplier must be integer>=1"

    model = nn.Sequential(
        nn.Conv1d(3, 16, 3, padding=1, stride=2),
        nn.SeparableConv1d(16, 24, 3, padding=1, stride=2, depth_multiplier=2),
        nn.SeparableConv1d(24, 32, 3, padding=1, stride=2),
        nn.SeparableConv1d(32, 40, 3, padding=1, stride=2, depth_multiplier=3),
        nn.SeparableConv1d(40, 48, 3, padding=1, stride=2),
    )

    output = model(input)

    assert output.shape == torch.Size([10, 48, 7])


def test_model2d():
    input = torch.randn(10, 3, 224, 224)

    error_message = ""

    try:
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, stride=2),
            nn.SeparableConv2d(16, 24, 3, padding=1, stride=2, depth_multiplier=2),
            nn.SeparableConv2d(24, 32, 3, padding=1, stride=2),
            nn.SeparableConv2d(32, 40, 3, padding=1, stride=2, depth_multiplier=0.5),
            nn.SeparableConv2d(40, 48, 3, padding=1, stride=2),
        )
    except Exception as e:
        error_message = str(e)

    assert error_message == "depth_multiplier must be integer>=1"

    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1, stride=2),
        nn.SeparableConv2d(16, 24, 3, padding=1, stride=2, depth_multiplier=2),
        nn.SeparableConv2d(24, 32, 3, padding=1, stride=2),
        nn.SeparableConv2d(32, 40, 3, padding=1, stride=2, depth_multiplier=3),
        nn.SeparableConv2d(40, 48, 3, padding=1, stride=2),
    )

    output = model(input)

    assert output.shape == torch.Size([10, 48, 7, 7])


def test_model3d():
    input = torch.randn(10, 3, 224, 224, 224)

    error_message = ""

    try:
        model = nn.Sequential(
            nn.Conv3d(3, 16, 3, padding=1, stride=2),
            nn.SeparableConv3d(16, 24, 3, padding=1, stride=2, depth_multiplier=2),
            nn.SeparableConv3d(24, 32, 3, padding=1, stride=2),
            nn.SeparableConv3d(32, 40, 3, padding=1, stride=2, depth_multiplier=0.5),
            nn.SeparableConv3d(40, 48, 3, padding=1, stride=2),
        )
    except Exception as e:
        error_message = str(e)

    assert error_message == "depth_multiplier must be integer>=1"

    model = nn.Sequential(
        nn.Conv3d(3, 16, 3, padding=1, stride=2),
        nn.SeparableConv3d(16, 24, 3, padding=1, stride=2, depth_multiplier=2),
        nn.SeparableConv3d(24, 32, 3, padding=1, stride=2),
        nn.SeparableConv3d(32, 40, 3, padding=1, stride=2, depth_multiplier=3),
        nn.SeparableConv3d(40, 48, 3, padding=1, stride=2),
    )

    output = model(input)

    assert output.shape == torch.Size([10, 48, 7, 7, 7])


if __name__ == "__main__":
    pytest.main([__file__])
