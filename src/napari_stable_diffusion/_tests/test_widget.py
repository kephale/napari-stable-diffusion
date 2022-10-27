import numpy as np

from napari_stable_diffusion import (
    StableDiffusionWidget,
    StableDiffusionImg2ImgWidget,
    StableDiffusionInpaintWidget,
)


# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
def test_example_text2img(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))

    # create our widget, passing in the viewer
    my_widget = StableDiffusionWidget(viewer)

    assert my_widget is not None


def test_example_img2img(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))

    # create our widget, passing in the viewer
    my_widget = StableDiffusionImg2ImgWidget(viewer)

    assert my_widget is not None


def test_example_inpaint(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))

    # create our widget, passing in the viewer
    my_widget = StableDiffusionInpaintWidget(viewer)

    assert my_widget is not None
