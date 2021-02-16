import numpy as np
from gunpowder.array import ArrayKey, Array
from gunpowder.array_spec import ArraySpec
from gunpowder.ext import torch
from gunpowder.nodes.generic_predict import GenericPredict

import logging
from typing import Dict, Union

logger = logging.getLogger(__name__)


class Predict(GenericPredict):
    """Torch implementation of :class:`gunpowder.nodes.Predict`.

    Args:

        model (subclass of ``torch.nn.Module``):

            The model to use for prediction.

        inputs (``dict``, ``string`` -> :class:`ArrayKey`):

            Dictionary from the names of input tensors (argument names of the
            ``forward`` method) in the model to array keys.

        outputs (``dict``, ``string`` or ``int`` -> :class:`ArrayKey`):

            Dictionary from the names of tensors in the network to array
            keys. If the key is a string, the tensor will be retrieved
            by checking the model for an attribute with the key as its name.
            If the key is an integer, it is interpreted as a tuple index of
            the outputs of the network.
            New arrays will be generated by this node for each entry (if
            requested downstream).

        array_specs (``dict``, :class:`ArrayKey` -> :class:`ArraySpec`, optional):

            Used to set the specs of generated arrays (``outputs``). This is
            useful to set the ``voxel_size``, for example, if they differ from
            the voxel size of the input arrays. Only fields that are not
            ``None`` in the given :class:`ArraySpec` will be used.

        checkpoint: (``string``, optional):

            An optional path to the saved parameters for your torch module.
            These will be loaded and used for prediction if provided.

        device (``string``, optional):

            Which device to use for prediction (``"cpu"`` or ``"cuda"``).
            Default is ``"cuda"``, which falls back to CPU if CUDA is not
            available.

        spawn_subprocess (bool, optional): Whether to run ``predict`` in a
            separate process. Default is false.
    """

    def __init__(
        self,
        model,
        inputs: Dict[str, ArrayKey],
        outputs: Dict[Union[str, int], ArrayKey],
        array_specs: Dict[ArrayKey, ArraySpec] = None,
        checkpoint: str = None,
        device="cuda",
        spawn_subprocess=False
    ):

        self.array_specs = array_specs if array_specs is not None else {}

        if model.training:
            logger.warning(
                "Model is in training mode during prediction. "
                "Consider using model.eval()"
            )

        super(Predict, self).__init__(
            inputs,
            outputs,
            array_specs,
            spawn_subprocess=spawn_subprocess)

        self.device_string = device
        self.device = None  # to be set in start()
        self.model = model
        self.checkpoint = checkpoint

        self.intermediate_layers = {}
        self.register_hooks()

    def start(self):

        self.use_cuda = (
            torch.cuda.is_available() and
            self.device_string == "cuda")
        logger.info(f"Predicting on {'gpu' if self.use_cuda else 'cpu'}")
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        try:
            self.model = self.model.to(self.device)
        except RuntimeError as e:
            raise RuntimeError(
                "Failed to move model to device. If you are using a child process "
                "to run your model, maybe you already initialized CUDA by sending "
                "your model to device in the main process."
            ) from e

        if self.checkpoint is not None:
            checkpoint = torch.load(self.checkpoint, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict()

    def predict(self, batch, request):
        inputs = self.get_inputs(batch)
        with torch.no_grad():
            out = self.model.forward(**inputs)
        outputs = self.get_outputs(out, request)
        self.update_batch(batch, request, outputs)

    def get_inputs(self, batch):
        model_inputs = {
            key: torch.as_tensor(np.ascontiguousarray(batch[value].data)).to(
                device=self.device, non_blocking=True)
            for key, value in self.inputs.items()
        }
        return model_inputs

    def register_hooks(self):
        for key in self.outputs:
            if isinstance(key, str):
                layer = getattr(self.model, key)
                layer.register_forward_hook(self.create_hook(key))

    def create_hook(self, key):
        def save_layer(module, input, output):
            self.intermediate_layers[key] = output

        return save_layer

    def get_outputs(self, module_out, request):
        outputs = {}
        if isinstance(module_out, tuple):
            module_outs = module_out
        else:
            module_outs = (module_out,)
        for key, value in self.outputs.items():
            if value in request:
                if isinstance(key, str):
                    outputs[value] = self.intermediate_layers[key]
                elif isinstance(key, int):
                    outputs[value] = module_outs[key]
        return outputs

    def update_batch(self, batch, request, requested_outputs):
        for array_key, tensor in requested_outputs.items():
            spec = self.spec[array_key].copy()
            spec.roi = request[array_key].roi
            batch.arrays[array_key] = Array(tensor.detach().cpu().numpy(), spec)

    def stop(self):
        pass
