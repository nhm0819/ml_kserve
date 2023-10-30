import argparse
from typing import Dict, Union, Tuple
from torchvision import transforms
import numpy as np
import kserve.constants
from kserve import model_server, Model, ModelServer, InferInput, InferRequest
import os
import requests
import io
from PIL import Image
import logging
import base64
import grpc
from kserve.protocol.grpc import grpc_predict_v2_pb2_grpc

# gcs connect
from google.cloud import storage
storage_client = storage.Client.from_service_account_json("/secrets/platform_account.json")
# storage_client = storage.Client.from_service_account_json("dockerfiles/image-retrieval/platform_account.json")

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class ImageRetrieval(Model):
    def __init__(self, name: str, predictor_host: str, protocol: str):
        super().__init__(name)
        self.predictor_host = predictor_host
        self.protocol = protocol
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
        # self.storage_client = storage.Client.from_service_account_json("/secrets/service_account.json")
        self.ready = True

    def image_transform(self, data: Union[str, bytes, bytearray, list]):
        if data.startswith("gs://"):
            print(data)

            # blob name
            split_path = data.split('/')
            bucket_name = split_path[2]
            bucket = storage_client.get_bucket(bucket_name)

            blob_name = '/'.join(split_path[3:])
            blob = bucket.blob(blob_name=blob_name)
            if not blob.exists():
                raise f"Cannot find '{data}', Check gcs url"

            img_bytes = blob.download_as_bytes()
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            prep_image = self.transform(image)

            return prep_image.numpy()


        elif data.startswith("http") or data.startswith("https"):
            split_path = data.split('/')
            bucket_name = split_path[3]
            bucket = storage_client.get_bucket(bucket_name)

            blob_name = '/'.join(split_path[4:])
            blob = bucket.blob(blob_name=blob_name)
            if not blob.exists():
                raise f"Cannot find '{data}', Check gcs url"

            img_bytes = blob.download_as_bytes()
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            prep_image = self.transform(image)

            return prep_image.numpy()

        if isinstance(data, str):
            # if the image is a string of bytesarray.
            image_byte = base64.b64decode(data)
        else:
            image_byte = data

        # If the image is sent as bytesarray
        if isinstance(image_byte, (bytearray, bytes)):
            image = Image.open(io.BytesIO(image_byte))
            prep_image = self.transform(image)
            # prep_image = np.array(prep_image)

        elif isinstance(image_byte, list):
            # if the image is a list
            image = np.asarray(image_byte)
            image = Image.fromarray(image)
            prep_image = self.transform(image)
            # prep_image = np.array(prep_image)

        return prep_image.numpy()

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
            input_tensors = [self.image_transform(data) for data in payload.inputs[0].data]
        else:
            input_tensors = [self.image_transform(instance["image"]["b64"]) for instance in payload["instances"]]
        input_tensors = np.asarray(input_tensors, dtype=np.float32)
        infer_inputs = InferInput(name="image_feature", datatype='FP32', shape=list(input_tensors.shape),
                                  data=input_tensors)

        request_id = str(payload.id) if isinstance(payload.id, str) else "N.A."
        infer_request = InferRequest(model_name=self.name, infer_inputs=[infer_inputs], request_id=request_id)

        return infer_request


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[model_server.parser])
    parser.add_argument(
        "--predictor_host", help="The URL for the model predict function", required=True
    )
    parser.add_argument(
        "--protocol", help="The protocol for the predictor", default="v2"
    )
    parser.add_argument(
        "--model_name", help="The name that the model is served under.", default="image-retrieval"
    )
    args, _ = parser.parse_known_args()

    model = ImageRetrieval(args.model_name, predictor_host=args.predictor_host,
                           protocol=args.protocol)
    ModelServer().start([model])
