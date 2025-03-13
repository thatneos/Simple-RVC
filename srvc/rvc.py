from multiprocessing import cpu_count
from pathlib import Path
import os
import torch
from fairseq import checkpoint_utils
from scipy.io import wavfile
import yt_dlp
from srvc.pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from srvc.my_utils import load_audio
from srvc.pipeline import VC

BASE_DIR = Path(os.getcwd())





class MyLogger:
    def debug(self, msg):
        print("[CUSTOM DEBUG] " + msg)

    def info(self, msg):
        print("[CUSTOM INFO] " + msg)

    def warning(self, msg):
        print("[CUSTOM WARNING] " + msg)

    def error(self, msg):
        print("[CUSTOM ERROR] " + msg)

def ytdl(url: str) -> None:
    """
    Download audio from a YouTube video using the yt-dlp Python API.

    This function uses yt-dlp options to download the best available audio,
    extract it, and convert it to a WAV file. It also uses cookies from 'srvc/cnfg.txt'
    and a custom logger for logging output.

    Args:
        url (str): The URL of the YouTube video.
    """
    ydl_opts = {
        'format': 'bestaudio',
        'cookiefile': 'srvc/cnfg.txt',
        'logger': MyLogger(),  # Use our custom logger
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '0',  # '0' indicates best quality
        }]
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


class RVCUtil:
    def __init__(self, device, is_half, config):
        """
        Initializes the utility with a device, half-precision flag, and a configuration object.
        """
        self.device = device
        self.is_half = is_half
        self.config = config

    def load_hubert(self, model_path):
        """
        Loads a Hubert model from the given model_path,
        moves it to the configured device, converts to half/float as needed,
        and sets it in evaluation mode.
        """
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([model_path], suffix='')
        hubert = models[0].to(self.device)

        hubert = hubert.half() if self.is_half else hubert.float()
        hubert.eval()
        return hubert

    def get_vc(self, model_path):
        """
        Loads a VC model from model_path and returns a tuple containing:
         - the checkpoint (cpt)
         - version string ("v1" or "v2")
         - the VC model (net_g)
         - target sampling rate (tgt_sr)
         - the VC pipeline instance (vc)
        """
        cpt = torch.load(model_path, map_location='cpu')
        if "config" not in cpt or "weight" not in cpt:
            raise ValueError(
                f'Incorrect format for {model_path}. Use a voice model trained using RVC v2 instead.'
            )

        tgt_sr = cpt["config"][-1]
        # Update config with the speaker embedding size
        cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
        if_f0 = cpt.get("f0", 1)
        version = cpt.get("version", "v1")

        if version == "v1":
            if if_f0 == 1:
                net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=self.is_half)
            else:
                net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
        elif version == "v2":
            if if_f0 == 1:
                net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=self.is_half)
            else:
                net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])

        # Remove the unused encoder part and load weights
        del net_g.enc_q
        print(net_g.load_state_dict(cpt["weight"], strict=False))
        net_g.eval().to(self.device)

        net_g = net_g.half() if self.is_half else net_g.float()

        vc = VC(tgt_sr, self.config)
        return cpt, version, net_g, tgt_sr, vc

    def rvc_infer(
        self,
        index_path,
        index_rate,
        input_path,
        output_path,
        pitch_change,
        f0_method,
        cpt,
        version,
        net_g,
        filter_radius,
        tgt_sr,
        rms_mix_rate,
        protect,
        crepe_hop_length,
        vc,
        hubert_model,
        f0autotune,
        f0_min=50,
        f0_max=1100
    ):
        """
        Loads an audio file from input_path, performs voice conversion using the provided
        models and parameters, and writes the output to output_path.
        """
        audio = load_audio(input_path, 16000)
        times = [0, 0, 0]
        if_f0 = cpt.get("f0", 1)
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            0,
            audio,
            input_path,
            times,
            pitch_change,
            f0_method,
            index_path,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            0,
            rms_mix_rate,
            version,
            protect,
            crepe_hop_length,
            f0autotune,
            f0_file=None,
            f0_min=f0_min,
            f0_max=f0_max
        )
        wavfile.write(output_path, tgt_sr, audio_opt)
