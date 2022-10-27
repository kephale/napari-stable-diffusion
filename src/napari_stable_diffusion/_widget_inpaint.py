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
    QCheckBox,
    QVBoxLayout,
    QLabel,
    QPlainTextEdit,
)

from magicgui.widgets import create_widget

from PIL import Image
import numpy as np

import os

import torch
from diffusers import StableDiffusionInpaintPipeline

import napari


class StableDiffusionInpaintWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        # Textbox for entering prompt
        self.prompt_textbox = QPlainTextEdit(self)

        # Number of output images
        self.gallery_size = QSpinBox(self)
        self.gallery_size.setMinimum(1)
        self.gallery_size.setValue(9)

        # Width and height
        self.width_input = QSpinBox(self)
        self.width_input.setMinimum(1)
        self.width_input.setMaximum(2**31 - 1)
        # Overflows if larger than this maximum
        self.width_input.setValue(512)

        self.height_input = QSpinBox(self)
        self.height_input.setMinimum(1)
        self.height_input.setMaximum(2**31 - 1)
        self.height_input.setValue(512)

        # Select devices:
        # CPU is always available
        available_devices = ["cpu"]
        # Add 'mps' for M1
        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
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

        # Not Safe For Work button
        self.nsfw_button = QCheckBox(self)
        self.nsfw_button.setCheckState(True)

        btn = QPushButton("Run")
        btn.clicked.connect(self._on_click)

        # Layout and labels
        self.setLayout(QVBoxLayout())

        # Image selection widget
        self._image_layers = create_widget(annotation=napari.layers.Image)
        self.layout().addWidget(QLabel("Image"))
        self.layout().addWidget(self._image_layers.native)

        # Mask selection widget
        self._mask_layers = create_widget(annotation=napari.layers.Image)
        self.layout().addWidget(QLabel("Mask"))
        self.layout().addWidget(self._mask_layers.native)

        label = QLabel(self)
        label.setText("Prompt")
        self.layout().addWidget(label)
        self.layout().addWidget(self.prompt_textbox)

        label = QLabel(self)
        label.setText("Number of images")
        self.layout().addWidget(label)
        self.layout().addWidget(self.gallery_size)

        label = QLabel(self)
        label.setText("Number of inference steps")
        self.layout().addWidget(label)
        self.layout().addWidget(self.num_inference_steps)

        label = QLabel(self)
        label.setText("Image width")
        self.layout().addWidget(label)
        self.layout().addWidget(self.width_input)

        label = QLabel(self)
        label.setText("Image height")
        self.layout().addWidget(label)
        self.layout().addWidget(self.height_input)

        label = QLabel(self)
        label.setText("Enable Not Safe For Work mode")
        self.layout().addWidget(label)
        self.layout().addWidget(self.nsfw_button)

        label = QLabel(self)
        label.setText("Compute device")
        self.layout().addWidget(label)
        self.layout().addWidget(self.device_list)

        self.layout().addWidget(btn)

    def _on_click(self):
        # Has issues on mps and small GPUs
        # self.generate_images_batch()

        # worker = create_worker(self.generate_images_sequential)
        # worker.start()

        # TODO: Notify the user that things are processing

        self.generate_images_sequential()

    def generate_images_sequential(self):
        prompt = self.prompt_textbox.document().toPlainText()
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
        height = self.height_input.value()
        width = self.width_input.value()
        latents_shape = (batch_size, in_channels, height // 8, width // 8)

        # Load the pipeline
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            use_auth_token=MY_SECRET_TOKEN,
            height=height,
            width=width,
            num_inference_steps=self.num_inference_steps.value(),
        )
        pipe.to(device)

        # Fail if no image selected
        if self._image_layers.value is None:
            print("No image selected")
            return

        # Fail if no mask selected
        if self._image_layers.value is None:
            print("No mask selected")
            return

        # Get initial image
        init_image = Image.fromarray(self._image_layers.value.data).convert(
            "RGB"
        )
        init_image = init_image.resize((768, 512))

        # Get initial mask
        mask = Image.fromarray(self._mask_layers.value.data).convert("RGB")
        mask = mask.resize((768, 512))

        # Run the pipeline
        num_images = self.gallery_size.value()

        # Populate the gallery
        for gallery_id in range(num_images):
            # Generate our random latent space uniquely per image
            latents = torch.randn(
                latents_shape,
                generator=None,
                device=("cpu" if device == "mps" else device),
                #            dtype=torch.float16,
            )
            pipe.latents = latents
            pipe.to(device)

            # TODO add strength and guidance_scale to GUI
            image_list = pipe(
                prompt=[prompt],
                init_image=init_image,
                mask_image=mask,
                strength=0.75,
                guidance_scale=7.5,
            )

            # This is the SD output
            array = np.array(image_list.images[0])

            # Mask the output
            array = np.where(mask == 1, array, init_image)

            # If NSFW, then zero over image
            if image_list["nsfw_content_detected"][0]:
                array = np.zeros_like(array)

            # Empty GPU cache as we generate images
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.viewer.add_image(
                array, name=f"nsd_{prompt}-{gallery_id}", rgb=True
            )

        # Show gallery as grid
        self.viewer.grid.enabled = True
