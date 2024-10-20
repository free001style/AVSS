import os
from lipreading.utils import load_model
from lipreading.model import Lipreading


def get_video_model(config):
    tcn_options = {}
    densetcn_options = {'block_config': config.densetcn_options.densetcn_block_config,
                        'growth_rate_set': config.densetcn_options.densetcn_growth_rate_set,
                        'reduced_size': config.densetcn_options.densetcn_reduced_size,
                        'kernel_size_set': config.densetcn_options.densetcn_kernel_size_set,
                        'dilation_size_set': config.densetcn_options.densetcn_dilation_size_set,
                        'squeeze_excitation': config.densetcn_options.densetcn_se,
                        'dropout': config.densetcn_options.densetcn_dropout,
                        }
    model = Lipreading(modality=config.modality,
                       num_classes=config.num_classes,
                       tcn_options=tcn_options,
                       densetcn_options=densetcn_options,
                       backbone_type=config.backbone_type,
                       relu_type=config.relu_type,
                       width_mult=config.width_mult,
                       use_boundary=config.use_boundary,
                       extract_feats=config.extract_feats)
    assert os.path.isfile(config.model_path), \
        f"Model path does not exist. Path input: {config.model_path}"
    model = load_model(config.model_path, model, allow_size_mismatch=config.allow_size_mismatch)
    return model
