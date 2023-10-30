import torch.nn as nn
import torch.nn.functional as F


class ISCNet(nn.Module):

    def __init__(self, backbone, fc_dim=256, p=1.0, eps=1e-6):
        super().__init__()

        self.backbone = backbone

        self.fc = nn.Linear(self.backbone.feature_info.info[-1]['num_chs'], fc_dim, bias=False)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()
        self.p = float(p)
        self.eps = float(eps)

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def gem(self, x):
        x_size_2 = int(x.size(-2))
        x_size_1 = int(x.size(-1))
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x_size_2, x_size_1)).pow(1. / self.p)

    def forward(self, x):
        batch_size = int(x.shape[0])
        x = self.backbone(x)[-1]
        x = self.gem(x).view(batch_size, -1)
        x = self.fc(x)
        x = self.bn(x)
        x = F.normalize(x)
        return x


if __name__ == "__main__":

    # model structure
    import timm
    backbone = timm.create_model("tf_efficientnetv2_m.in21k_ft_in1k", features_only=True)
    model = ISCNet(backbone=backbone)

    # load weight
    import torch
    weight_file = "models/checkpoint_0009.pth.tar"
    state_dict = torch.load(weight_file, map_location='cpu')['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('module.'):
            state_dict[k[len('module.'):]] = state_dict[k]
            del state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    print("model load")

    # load torchscript
    model = torch.jit.load("models/image_retrieval.pt")
    model.eval()
    batch_size = 1
    width = 512
    height = 512
    input_shape = (batch_size, 3, width, height)
    input_names = ['image_feature']
    output_names = ['image_embedding']
    dynamic_axes = {'image_feature': {0: 'batch_size'},
                    'image_embedding': {0: 'batch_size'}}  # adding names for better debugging
    onnx_path = "models/image_retrieval/image_retrieval/model.onnx"

    def export_onnx_model(model, input_shape, onnx_path, input_names=None, output_names=None, dynamic_axes=None, opset_version=14):
        inputs = torch.ones(*input_shape)
        model(inputs)
        torch.onnx.export(model, inputs, onnx_path, input_names=input_names, output_names=output_names,
                          dynamic_axes=dynamic_axes, opset_version=opset_version, do_constant_folding=True)

    export_onnx_model(model, input_shape, onnx_path, input_names, output_names, dynamic_axes, opset_version=14)