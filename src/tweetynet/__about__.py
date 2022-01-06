import os.path

__all__ = [
    "__title__",
    "__summary__",
    "__uri__",
    "__version__",
    "__commit__",
    "__author__",
    "__email__",
    "__license__",
    "__copyright__",
]


try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = None


__title__ = "tweetynet"
__summary__ = "neural network that segments and labels birdsong and other animal vocalizations"
__uri__ = "https://github.com/yardencsGitHub/tweetynet"

__version__ = "0.7.0"

if base_dir is not None and os.path.exists(os.path.join(base_dir, ".commit")):
    with open(os.path.join(base_dir, ".commit")) as fp:
        __commit__ = fp.read().strip()
else:
    __commit__ = None

__author__ = "Yarden Cohen, David Nicholson"
__email__ = "yardencsgithub@users.noreply.github.com"

__license__ = "BSD"
__copyright__ = "2017-present %s" % __author__
