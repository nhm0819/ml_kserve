from kserve.protocol.grpc.grpc_predict_v2_pb2 import ModelInferResponse, ModelInferRequest
from kserve.protocol.grpc import grpc_predict_v2_pb2_grpc
from kserve import InferRequest, InferInput, InferenceServerClient
import json
import base64
import os
import grpc
import numpy as np
import cv2
from typing import Tuple

def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


if __name__=="__main__":
    INGRESS_HOST = '192.168.49.2'
    INGRESS_PORT = '32537'
    SERVICE_HOSTNAME = 'sam-image-encoder.default.example.com'
    MODEL_NAME = 'sam-image-encoder'


    image = cv2.imread('dog.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    newh, neww = get_preprocess_shape(image.shape[0], image.shape[1], 1024)
    image = cv2.resize(image, (neww, newh), interpolation=cv2.INTER_LINEAR)
    """Normalize pixel values and pad to a square input."""
    pixel_mean = [123.675, 116.28, 103.53]
    pixel_std = [58.395, 57.12, 57.375]

    # Normalize colors
    image = (image - pixel_mean) / pixel_std

    # Pad
    h, w = image.shape[:2]
    padh = 1024 - h
    padw = 1024 - w
    print("padh :", padh)
    print("padw :", padw)
    image = np.pad(image, ((0, padh), (0, padw), (0, 0)), mode='constant')
    print("padded.shape :", image.shape)

    # transpose
    image = image.transpose(2, 0, 1).astype(np.float32)
    input_tensor = np.asarray([image], dtype=np.float32)
    print("##############\n", input_tensor.shape)


    # client = InferenceServerClient(
    #     url=INGRESS_HOST + ":" + INGRESS_PORT,
    #     channel_args=(('grpc.ssl_target_name_override', SERVICE_HOSTNAME),))
    # infer_input = InferInput(name="image_feature", datatype='FP32', shape=list(input_tensor.shape))
    # infer_input.set_data_from_numpy(input_tensor=input_tensor, binary_data=True)
    # request = InferRequest(infer_inputs=[infer_input], model_name=MODEL_NAME)
    # res = client.infer(infer_request=request)
    # print(res)


    infer_inputs = InferInput(name="image_feature", datatype='FP32', shape=list(input_tensor.shape),
                              data=input_tensor)
    infer_inputs.set_data_from_numpy(input_tensor=input_tensor, binary_data=True)

    request_id = '0'
    infer_request = InferRequest(model_name="sam_image_encoder", infer_inputs=[infer_inputs], request_id=request_id)
    payload = infer_request.to_grpc()

    _channel = grpc.aio.insecure_channel('192.168.49.2:32537')
    _grpc_client_stub = grpc_predict_v2_pb2_grpc.GRPCInferenceServiceStub(_channel)

    async_result = _grpc_client_stub.ModelInfer(
        request=payload,
        # metadata=(('request_type', 'grpc_v2'),
        #           ('response_type', 'grpc_v2'),
        #           ('x-request-id', 'x-request-id'))
    )




