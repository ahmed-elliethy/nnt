# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Copyright © 2022 Ahmed Elliethy.
#
# All rights reserved.
#
# This software should be used, reproduced and modified only for informational and nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package)
#
import torch
import torch.nn as nn

from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names


class DnCnnNpPt(nn.Module):
    def __init__(self):
        super(DnCnnNpPt, self).__init__()
        self.model = torch.load("NeuralNoiseprintTransfer/Model/NoisePrint.pt")
        self.model.eval()

    # x holds the input tensor(image) that will be fed to each layer
    def forward(self, x):
        return self.model.forward(x)

    def get_output_at_node(self, x, nodes):
        model2 = create_feature_extractor(self.model, return_nodes=nodes)
        model2.eval()
        layers_out = model2(x)
        features = []
        for key, value in layers_out.items():
            return value
        return features

    def get_output_at_end(self, x):
        return self.get_output_at_node(x, ['dncnn.59'])  # 'dncnn.64.add'

    def get_output_at_first(self, x):
        vvv = get_graph_node_names(self.model)
        return self.get_output_at_node(x, ['dncnn.1.add'])
