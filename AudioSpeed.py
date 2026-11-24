import hashlib
import torch
import math
import io
import torchaudio

class NoCloseBytesIO(io.BytesIO):
    def real_close(self):
        # Call this if you genuinely want to close it later
        super().close()

    def close(self):
        # Prevent the underlying BytesIO from closing
        pass


class AudioSpeed:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "speed": ("FLOAT", {
                    "default": 1.5,
                    "step": 0.01,
                    "tooltip": "Speed. >1.0 slower. <1.0 faster",
                    }),
            },
            "optional": {
                "speed_type": (["torch-time-stretch", "TDHS"], {
                    "default": "TDHS",
                    "tooltip": "TDHS - Time-domain harmonic scaling. torch-time-stretch - torchaudio.transforms.TimeStretch.",  # noqa: E501
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    DESCRIPTION = "Change the speed of the audio"
    CATEGORY = "Audio"
    INPUT_IS_LIST = True

    FUNCTION = "time_shift"


    def time_shift_audiostretchy(self, audio, speed):
        from audiostretchy import AudioStretch
        from torchcodec.encoders import AudioEncoder
        rate = audio['sample_rate']
        waveform = audio['waveform']
        audio_stretch = AudioStretch()
        new_waveforms = []
        for channel in range(0, waveform.shape[0]):
            wav_buffer = io.BytesIO()
            encoder = AudioEncoder(waveform[0][channel], sample_rate=rate)
            encoder.to_file_like(wav_buffer, format="wav")
            nbytes = wav_buffer.getbuffer().nbytes
            wav_buffer.seek(0)  # rewind so it can be read from the start
            audio_stretch.open(file=wav_buffer, format="wav")
                                                                                                                                                                                # audiostretchy crashes in windows beyond these ranges even with double_range=True
            if speed == 0.5:
                speed = 0.5000001 # seg faults at 0.5 or 1

            double_range = False if(speed>0.5 and speed < 4) else True

            audio_stretch.stretch(
                ratio=speed,
                gap_ratio=1.2,
                upper_freq=333,
                lower_freq=55,
                buffer_ms=25,                                                                                                                                                       threshold_gap_db=-40,
                double_range=double_range,
                fast_detection=False,
                normal_detection=False,
            )
            # out_buffer = NoCloseBytesIO(b"\x00" * math.ceil(nbytes * speed))
            out_buffer = NoCloseBytesIO()

            out_buffer.seek(0)
            audio_stretch.save(file=out_buffer, format="wav")
            out_buffer.seek(0)  # rewind so it can be read from the start

            waveform, sample_rate = torchaudio.load(out_buffer)
            out_buffer.real_close()

            new_waveforms.append(waveform[0])

        new_waveform = torch.stack(new_waveforms)
        new_waveform = torch.stack([new_waveform])

        return {"waveform": new_waveform, "sample_rate": rate}

    def time_shift_torch_ts(self, audio, speed):
        import torch_time_stretch
        rate = audio['sample_rate']
        waveform = audio['waveform']

        new_waveform = torch_time_stretch.time_stretch(
            waveform,
            torch_time_stretch.Fraction(math.floor(speed*100), 100),
            rate
        )

        return {"waveform": new_waveform, "sample_rate": rate}

    def time_shift(
        self,
        audio,
        speed,
        speed_type="torch-time-stretch",
    ):
        new_audios = []

        if not isinstance(speed, list):
            speed = [speed] * len(audio)
        if all(x == 1 for x in speed):
            return (audio[0], )

        i = 0

        if isinstance(speed_type, list) and len(speed_type)==1:
            speed_type = speed_type[0]  # need this because we are using INPUT_IS_LIST

        for a in audio:
            if speed_type == "torch-time-stretch" or speed_type is None:
                new_audio = self.time_shift_torch_ts(a, speed[i])
            elif speed_type == "TDHS":
                new_audio = self.time_shift_audiostretchy(a, speed[i])
            else:
                raise ValueError(f"AudioSpeed: Unknown speed_type: {speed_type}")
            i += 1

            new_audios.append(new_audio)
        return (new_audio, )

    @classmethod
    def IS_CHANGED(
        s,
        **kwargs
    ):
        m = hashlib.sha256()
        keys = sorted(kwargs.keys())
        for key in keys:
            m.update(kwargs[key])

        return m.digest().hex()
