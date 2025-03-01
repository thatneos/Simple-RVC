# Simple-RVC


![counter](https://counter.seku.su/cmoe?name=demo&theme=mb)


Simple RVC Inference

# How to use

## Initialize configuration (using your provided Config class)
```
config = Config(device="cuda:0", is_half=True)
```
## Create an instance of the RVC utility
```

from srvc import RVCUtil


rvc_util = RVCUtil(device="cuda:0", is_half=True, config=config)
```
## Load the Hubert model

```
hubert_model = rvc_util.load_hubert("path/to/hubert_model.pt")
```
## Load the voice conversion model

```
cpt, version, net_g, tgt_sr, vc = rvc_util.get_vc("path/to/vc_model.pt")
```
## Run inference with desired parameters

```
rvc_util.rvc_infer(
    index_path="path/to/index_file",
    index_rate=0.5,
    input_path="path/to/input.wav",
    output_path="path/to/output.wav",
    pitch_change=0.0,
    f0_method="rmvpe",  # example: 'rmvpe' or 'fcpe'
    cpt=cpt,
    version=version,
    net_g=net_g,
    filter_radius=3,
    tgt_sr=tgt_sr,
    rms_mix_rate=0.33,
    protect=0.33,
    crepe_hop_length=160,
    vc=vc,
    hubert_model=hubert_model,
    f0autotune=False
)
```
