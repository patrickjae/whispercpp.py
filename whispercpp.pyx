#!python
# cython: language_level=3

import ffmpeg
import numpy as np
import requests
import os
import json
from pathlib import Path

MODELS_DIR = os.environ.get('MODELS_DIR', str(Path('~/.ggml-models').expanduser()))

cimport numpy as cnp

cdef int SAMPLE_RATE = 16000
cdef char* TEST_FILE = 'test.wav'
cdef char* MODEL = 'tiny'
cdef bytes l_b = os.environ.get('TARGET_LANGUAGE', 'en').encode('utf-8')
cdef char* LANGUAGE = l_b
cdef int N_THREADS = os.cpu_count()

MODELS = {
    'ggml-tiny.bin': 'https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin',
    'ggml-base.bin': 'https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-base.bin',
    'ggml-small.bin': 'https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-small.bin',
    'ggml-medium.bin': 'https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin',
    'ggml-large.bin': 'https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-large.bin',
}

def model_exists(target_dir, model_file):
    return os.path.exists(Path(target_dir).joinpath(model_file))

def download_model(target_dir, model_file):
    if model_exists(target_dir, model_file):
        return

    print(f'Downloading {model_file}...')
    url = MODELS[model_file]
    r = requests.get(url, allow_redirects=True)
    os.makedirs(target_dir, exist_ok=True)
    with open(Path(target_dir).joinpath(model_file), 'wb') as f:
        f.write(r.content)


cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] load_audio(bytes file, int sr = SAMPLE_RATE):
    try:
        out = (
            ffmpeg.input(file, threads=0)
            .output(
                "-", format="s16le",
                acodec="pcm_s16le",
                ac=1, ar=sr
            )
            .run(
                cmd=["ffmpeg", "-nostdin"],
                capture_stdout=True,
                capture_stderr=True
            )
        )[0]
    except:
        raise RuntimeError(f"File '{file}' not found")

    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] frames = (
        np.frombuffer(out, np.int16)
        .flatten()
        .astype(np.float32)
    ) / pow(2, 15)

    return frames

cdef whisper_full_params default_params() nogil:
    cdef whisper_sampling_strategy strategy = whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY
    if BEAM_SIZE > 1:
        strategy = whisper_sampling_strategy.WHISPER_SAMPLING_BEAM_SEARCH
    cdef whisper_full_params params = whisper_full_default_params(
        strategy
    )
    n_threads = N_THREADS
    params.translate = False
    params.print_progress = False
    params.print_realtime = False
    params.language = <const char *> LANGUAGE
    params.beam_search.beam_size = BEAM_SIZE
    params.beam_search.patience = PATIENCE
    return params


cdef class Whisper:
    cdef whisper_context * ctx
    cdef whisper_full_params params

    def __init__(self, model_path=None, model_dir=MODELS_DIR, model_type=MODEL):
        if not model_path or not os.path.isfile(model_path):
            model_fullname = f'ggml-{model_type}.bin'.encode('utf8')
            print(f'model path not specified or model file not present, downloading {model_type} model (filename: {model_fullname.decode()}) to {model_dir}')
            download_model(model_dir, model_fullname)
            model_path = Path(model_dir).joinpath(model_fullname)
        cdef bytes model_b = str(model_path).encode('utf8')
        self.ctx = whisper_init_from_file(model_b)
        whisper_print_system_info()
        self.params = default_params()

    def __dealloc__(self):
        whisper_free(self.ctx)

    def transcribe(self, filename=TEST_FILE, result_format='json'):
        cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] frames = load_audio(<bytes>filename)
        status = whisper_full(self.ctx, self.params, &frames[0], len(frames))
        if status != 0:
            raise RuntimeError
        return self.extract_result(result_format)

    def transcribe_from_frames(self, cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] frames, result_format='json'):
        print("Transcribing...")
        status = whisper_full(self.ctx, self.params, &frames[0], len(frames))
        if status != 0:
            raise RuntimeError
        return self.extract_result(result_format)
        
    
    def extract_result(self, result_format):
        cdef int n_segments = whisper_full_n_segments(self.ctx)
        cdef int n_tokens
        if result_format == 'text':
            return [
                whisper_full_get_segment_text(self.ctx, i).decode().strip() for i in range(n_segments)
            ]
        elif result_format == 'json':
            result = {
                'systeminfo': whisper_print_system_info().decode(),
                'model': {
                    'type': whisper_model_type_readable(self.ctx),
                    'multilingual': whisper_is_multilingual(self.ctx),
                    'vocab': whisper_model_n_vocab(self.ctx),
                    'audio': {
                        'ctx': whisper_model_n_audio_ctx(self.ctx),
                        'state': whisper_model_n_audio_state(self.ctx),
                        'head': whisper_model_n_audio_head(self.ctx),
                        'layer': whisper_model_n_audio_layer(self.ctx)
                    },
                    'text': {
                        'ctx': whisper_model_n_text_ctx(self.ctx),
                        'state': whisper_model_n_text_state(self.ctx),
                        'head': whisper_model_n_text_head(self.ctx),
                        'layer': whisper_model_n_text_layer(self.ctx)
                    },
                    'mels': whisper_model_n_mels(self.ctx),
                    'f16': whisper_model_f16(self.ctx),
                },
                'params': {
                    'model': MODEL.decode(),
                    'language': whisper_lang_str(whisper_lang_id(self.params.language)).decode(),
                    'translate': self.params.translate
                },
                'result': {
                    'language': whisper_lang_str(whisper_lang_id(self.params.language)).decode()
                },
                'transcription': {
                    'text': ' '.join([whisper_full_get_segment_text(self.ctx, i).decode().strip() for i in range(n_segments)]),
                    'segments': []
                }
            }
            for i in range(n_segments):
                n_tokens = whisper_full_n_tokens(self.ctx, i)
                av_log_prob = np.mean([np.log(whisper_full_get_token_p(self.ctx, i, j)) for j in range(n_tokens)])
                s = {
                    'id': i,
                    'start': whisper_full_get_segment_t0(self.ctx, i),
                    'end': whisper_full_get_segment_t1(self.ctx, i),
                    'text': whisper_full_get_segment_text(self.ctx, i).decode().strip(),
                    'tokens': [
                        whisper_full_get_token_id(self.ctx, i, j) for j in range(n_tokens)
                    ],
                    'avg_logprob': av_log_prob
                }
                result['transcription']['segments'].append(s)
            return result
        else:
            raise RuntimeError(f'Unknown result type {result_format}')