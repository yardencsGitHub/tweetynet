import sys

import nox


IS_MACOS = sys.platform == "darwin"


@nox.session
@nox.parametrize("torch",
                 [
                     "1.7.1",
                     "1.8.0",
                     "1.8.1",
                     "1.9.0",
                     "1.10.0",
                 ]
                 )
def tests(session, torch):
    torch_string = f"torch=={torch}"
    if not IS_MACOS:
        torch_string += "+cpu"
        torch_string = [torch_string, '-f', 'https://download.pytorch.org/whl/torch_stable.html']
    else:
        torch_string = [torch_string]  # wrap in list so we can unpack with * even if only one item
    session.install(*torch_string)
    session.install("vak>=0.4.0b5")
    session.install(".")
    session.install("pytest")
    session.run("pytest")
