"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from qtpy.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QWidget,
    QComboBox,
    QLineEdit,
    QSpinBox,
)

# import torch
# from torch import autocast // only for GPU

import numpy as np

import os

import torch
from diffusers import StableDiffusionPipeline

# from diffusers import StableDiffusionImg2ImgPipeline

if TYPE_CHECKING:
    import napari


class StableDiffusionWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        # Textbox for entering prompt
        self.prompt_textbox = QLineEdit(self)

        # Number of output images
        self.gallery_size = QSpinBox(self)
        self.gallery_size.setMinimum(1)
        self.gallery_size.setValue(9)

        # Select devices:
        # CPU is always available
        available_devices = ["cpu"]
        # Add 'mps' for M1
        if torch.backends.mps.is_available():
            available_devices += ["mps"]
        # Add 'cuda' for nvidia cards
        if torch.cuda.is_available():
            available_devices += [
                f"cuda:{id}" for id in range(torch.cuda.device_count())
            ]

        self.device_list = QComboBox(self)
        self.device_list.addItems(available_devices)

        self.num_inference_steps = QSpinBox(self)
        self.num_inference_steps.setMinimum(1)
        self.num_inference_steps.setValue(50)

        btn = QPushButton("Run")
        btn.clicked.connect(self._on_click)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(self.prompt_textbox)
        self.layout().addWidget(self.device_list)
        self.layout().addWidget(btn)

    def _on_click(self):
        prompt = self.prompt_textbox.text()
        print(f"Prompt is {prompt}")

        # Get the device: cpu or gpu
        device = self.device_list.currentText()

        # Get huggingface token from environment variable. Generate at HF
        MY_SECRET_TOKEN = (
            os.environ.get("HF_TOKEN_SD")
            if "HF_TOKEN_SD" in os.environ
            else None
        )

        # Pre-generate the latents to ensure correct dtype
        batch_size = len(prompt)
        in_channels = 3
        height = 512
        width = 512
        latents_shape = (batch_size, in_channels, height // 8, width // 8)
        latents = torch.randn(
            latents_shape,
            generator=None,
            device=("cpu" if device == "mps" else device),
            dtype=torch.float16,
        )

        # Load the pipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            use_auth_token=MY_SECRET_TOKEN,
            torch_dtype=torch.float16,
            revision="fp16",
            output_type="ndarray",
            latents=latents,
            height=height,
            width=width,
            num_inference_steps=self.num_inference_steps.value(),
        )
        pipe.to(device)

        # Run the pipeline
        num_images = self.gallery_size.value()
        image_list = pipe([prompt] * num_images)

        # Populate the gallery
        for gallery_id in range(num_images):
            # For PIL outputs
            # array = np.array(image_list.images[gallery_id])

            # For ndarray outputs
            array = image_list.images[gallery_id]

            # Empty GPU cache as we generate images
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.viewer.add_image(array, rgb=True)

        # Show gallery as grid
        self.viewer.grid.enabled = True
