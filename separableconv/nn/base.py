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


import torch.nn as nn


class _SeparableConv(nn.Module):
    def __init__(self, *args, **kwargs):
        super(_SeparableConv, self).__init__()

        self.dwconv = None
        self.dwconv_normalization = None
        self.dwconv_activation = None

        self.pwconv = None
        self.pwconv_normalization = None
        self.pwconv_activation = None

    def forward(self, x):
        assert self.dwconv is not None and self.pwconv is not None, (
            "Depthwise Convolution and/or Pointwise Convolution is/are not implemented"
            " yet."
        )

        x = self.dwconv(x)

        if self.dwconv_normalization is not None:
            x = self.dwconv_normalization(x)

        if self.dwconv_activation is not None:
            x = self.dwconv_activation(x)

        x = self.pwconv(x)

        if self.pwconv_normalization is not None:
            x = self.pwconv_normalization(x)

        if self.dwconv_activation is not None:
            x = self.pwconv_activation(x)

        return x
