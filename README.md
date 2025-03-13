
# Simple RVC

![counter](https://counter.seku.su/cmoe?name=demo&theme=mb)

A Simple RVC Inference

## How to Use

### Initialize Configuration

First, initialize the configuration using your provided `Config` class:

```python
config = Config(device="cuda:0", is_half=True)
```

### Create an Instance of the RVC Utility

Next, create an instance of the `RVCUtil`:

```python
from srvc import RVCUtil

rvc_util = RVCUtil(device="cuda:0", is_half=True, config=config)
```

### Load the Hubert Model

Load the Hubert model with the following command:

```python
hubert_model = rvc_util.load_hubert("path/to/hubert_model.pt")
```

### Load the Voice Conversion Model

Load the voice conversion model using:

```python
cpt, version, net_g, tgt_sr, vc = rvc_util.get_vc("path/to/vc_model.pth")
```

### Run Inference

Finally, run the inference with the desired parameters:

```python
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

### Notes

- Ensure that you replace the placeholder paths (e.g., `path/to/hubert_model.pth`) with the actual paths to your model files.
- Adjust the inference parameters according to your requirements.

This should make the README more structured and easier to follow.
