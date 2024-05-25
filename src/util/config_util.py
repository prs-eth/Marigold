# Author: Bingxin Ke
# Last modified: 2024-02-14

import omegaconf
from omegaconf import OmegaConf


def recursive_load_config(config_path: str) -> OmegaConf:
    conf = OmegaConf.load(config_path)

    output_conf = OmegaConf.create({})

    # Load base config. Later configs on the list will overwrite previous
    base_configs = conf.get("base_config", default_value=None)
    if base_configs is not None:
        assert isinstance(base_configs, omegaconf.listconfig.ListConfig)
        for _path in base_configs:
            assert (
                _path != config_path
            ), "Circulate merging, base_config should not include itself."
            _base_conf = recursive_load_config(_path)
            output_conf = OmegaConf.merge(output_conf, _base_conf)

    # Merge configs and overwrite values
    output_conf = OmegaConf.merge(output_conf, conf)

    return output_conf


def find_value_in_omegaconf(search_key, config):
    result_list = []

    if isinstance(config, omegaconf.DictConfig):
        for key, value in config.items():
            if key == search_key:
                result_list.append(value)
            elif isinstance(value, (omegaconf.DictConfig, omegaconf.ListConfig)):
                result_list.extend(find_value_in_omegaconf(search_key, value))
    elif isinstance(config, omegaconf.ListConfig):
        for item in config:
            if isinstance(item, (omegaconf.DictConfig, omegaconf.ListConfig)):
                result_list.extend(find_value_in_omegaconf(search_key, item))

    return result_list


if "__main__" == __name__:
    conf = recursive_load_config("config/train_base.yaml")
    print(OmegaConf.to_yaml(conf))
