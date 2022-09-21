# napari-stable-diffusion

[![License BSD-3](https://img.shields.io/pypi/l/napari-stable-diffusion.svg?color=green)](https://github.com/kephale/napari-stable-diffusion/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-stable-diffusion.svg?color=green)](https://pypi.org/project/napari-stable-diffusion)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-stable-diffusion.svg?color=green)](https://python.org)
[![tests](https://github.com/kephale/napari-stable-diffusion/workflows/tests/badge.svg)](https://github.com/kephale/napari-stable-diffusion/actions)
[![codecov](https://codecov.io/gh/kephale/napari-stable-diffusion/branch/main/graph/badge.svg)](https://codecov.io/gh/kephale/napari-stable-diffusion)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-stable-diffusion)](https://napari-hub.org/plugins/napari-stable-diffusion)

A demo of stable diffusion in napari.

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

![demo image of napari-stable-diffusion of the prompt "a unicorn and a dinosaur eating cookies and drinking tea"](./napari_stable_diffusion_demo.png)

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `napari-stable-diffusion` via [pip]:

    pip install napari-stable-diffusion

To install latest development version :

    pip install git+https://github.com/kephale/napari-stable-diffusion.git

You will also need to sign up with HuggingFace and [generate an access
token](https://huggingface.co/docs/hub/security-tokens) to get access to the
Stable Diffusion model we use.

When you have generated your access token you can either permanently
set the `HF_TOKEN_SD` environment variable in your `.bashrc` or whichever file
your OS uses, or you can include it on the command line

```
HF_TOKEN_SD="hf_aaaAaaaasdadsadsaoaoaoasoidijo" napari
```

For more information on the Stable Diffusion model itself, please see https://huggingface.co/CompVis/stable-diffusion-v1-4.

### Apple M1 specific instructions

To utilize the M1 GPU, the nightly version of PyTorch needs to be
installed. Consider using `conda` or `mamba` like this:

```
mamba create -c pytorch-nightly -n napari-stable-diffusion python=3.9 pip pyqt pytorch torchvision
pip install git+https://github.com/kephale/napari-stable-diffusion.git
```

## Next steps

- Image 2 Image support
- Inpainting support

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-stable-diffusion" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/kephale/napari-stable-diffusion/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
