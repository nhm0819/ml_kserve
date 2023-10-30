# Copyright 2021 The KServe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from typing import Dict, Union, Tuple

import numpy

import cv2
from kserve.protocol.grpc.grpc_predict_v2_pb2 import ModelInferResponse, ModelInferRequest
from kserve import Model, ModelServer, model_server, InferInput, InferRequest, InferResponse
from kserve.model import PredictorProtocol
import kserve.constants
from kserve.protocol.grpc import grpc_predict_v2_pb2_grpc
import grpc

# import torch
# from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore

import logging
import base64

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)

def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)

def norm_pad_transpose(x: numpy.array, target_length: int=1024) -> numpy.array:
    """Normalize pixel values and pad to a square input."""
    pixel_mean = [123.675, 116.28, 103.53]
    pixel_std = [58.395, 57.12, 57.375]

    # Normalize colors
    x = (x - pixel_mean) / pixel_std

    # Pad
    h, w = x.shape[:2]
    padh = target_length - h
    padw = target_length - w
    x = numpy.pad(x, ((0, padh), (0, padw), (0, 0)), mode='constant')

    # transpose
    x = x.transpose(2, 0, 1)

    return x

def image_transform(model_name, data, target_length:int = 1024):
    """converts the input image of Bytes Array into Tensor
    Args:
        data: The input image bytes.
    Returns:
        numpy.array: Returns the numpy array after the image preprocessing.
    """

    byte_array = base64.b64decode(data)
    encoded_img = numpy.frombuffer(byte_array, dtype=numpy.uint8)
    image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR) # image = cv2.imread("images/dog.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    newh, neww = get_preprocess_shape(image.shape[0], image.shape[1], target_length)

    print("newh :", newh)
    print("neww :", neww)
    print("image.shape :", image.shape)
    # image = cv2.resize(image, (neww, newh), interpolation=cv2.INTER_LINEAR)
    image = numpy.array(resize(to_pil_image(image), (newh, neww)))
    print("resized.shape :", image.shape)
    image_array = norm_pad_transpose(image, target_length=target_length)

    return image_array


class ImageTransformer(Model):
    def __init__(self, name: str, predictor_host: str, protocol: str):
        super().__init__(name)
        self.predictor_host = predictor_host
        self.protocol = protocol
        self.ready = True

    @property
    def _grpc_client(self):
        options = [('grpc.max_message_length', 100 * 1024 * 1024),
                   ('grpc.max_send_message_length', 100 * 1024 * 1024),
                   ('grpc.max_receive_message_length', 100 * 1024 * 1024)]
        if self._grpc_client_stub is None:
            # requires appending ":80" to the predictor host for gRPC to work
            if ":" not in self.predictor_host:
                self.predictor_host = self.predictor_host + ":80"
            _channel = grpc.aio.insecure_channel(self.predictor_host, options=options)
            self._grpc_client_stub = grpc_predict_v2_pb2_grpc.GRPCInferenceServiceStub(_channel)
        return self._grpc_client_stub

    def preprocess(self, payload: Union[Dict, InferRequest], headers: Dict[str, str] = None) \
            -> Union[Dict, InferRequest]:
        if isinstance(payload, InferRequest):
            input_tensors = [image_transform(self.name, instance) for instance in payload.inputs[0].data]
        else:
            input_tensors = [image_transform(self.name, instance["image"]["b64"]) for instance in payload["instances"]]
        input_tensors = numpy.asarray(input_tensors, dtype=numpy.float32)
        infer_inputs = InferInput(name="image_feature", datatype='FP32', shape=list(input_tensors.shape),
                                  data=input_tensors)

        request_id = str(payload.id) if isinstance(payload.id, str) else "N.A."
        infer_request = InferRequest(model_name=self.name, infer_inputs=[infer_inputs], request_id=request_id)

        return infer_request


    async def _grpc_predict(self, payload: Union[ModelInferRequest, InferRequest], headers: Dict[str, str] = None) \
            -> ModelInferResponse:
        request_id = payload.id if payload.id else "N.A."
        if isinstance(payload, InferRequest):
            payload = payload.to_grpc()

        async_result = await self._grpc_client.ModelInfer(
            request=payload,
            timeout=self.timeout,
            metadata=(('request_type', 'grpc_v2'),
                      ('response_type', 'grpc_v2'),
                      ('x-request-id', request_id))
        )
        return async_result




if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[model_server.parser])
    parser.add_argument(
        "--predictor_host", help="The URL for the model predict function", required=True
    )
    parser.add_argument(
        "--protocol", help="The protocol for the predictor", default="v2"
    )
    parser.add_argument(
        "--model_name", help="The name that the model is served under.", default="sam_image_encoder"
    )
    args, _ = parser.parse_known_args()

    model = ImageTransformer(args.model_name, predictor_host=args.predictor_host,
                             protocol=args.protocol)
    ModelServer().start([model])
