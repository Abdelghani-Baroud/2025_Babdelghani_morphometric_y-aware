from hydra import initialize, compose

# assumes you're inside project_root or adjust config_path accordingly
with initialize(config_path="configs", version_base=None):
    cfg = compose(config_name="config")
    from omegaconf import OmegaConf
    print(OmegaConf.to_yaml(cfg))

# Now you can access anything from AGEPGS.yaml
print(cfg.label_debug)          # should print "thickness_volume"
print(cfg.label_names[0]) 