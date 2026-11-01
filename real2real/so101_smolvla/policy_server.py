"""Real-data policy server preserving the image geometry used during training.

The upstream async helper resizes images to config.image_features (256x256),
although training samples are 480x640 and SmolVLA later pads them to 512x512.
This wrapper preserves native camera dimensions so that SmolVLA applies the
same aspect-preserving transform as training. Weights/processors are unchanged.
Accepts native uint8 arrays (preview client) or lossless RGB PNG (real runner).
"""

from dataclasses import replace
import hashlib
import pickle
from pathlib import Path

import cv2
import numpy as np

from settings import load_settings
MODEL = Path(load_settings()['server_model_path'])
PIPELINE = 'native-rgb-v1'


def install_server_protocol(policy_server):
    """Identify this image pipeline and reuse weights between supervised trials."""
    original_ready = policy_server.PolicyServer.Ready
    original_setup = policy_server.PolicyServer.SendPolicyInstructions
    with (MODEL/'model.safetensors').open('rb') as weights:
        digest = hashlib.file_digest(weights, 'sha256').hexdigest()

    def ready(self, request, context):
        result = original_ready(self, request, context)
        self.last_processed_obs = None
        context.set_trailing_metadata((('real-image-pipeline', PIPELINE), ('real-model-sha256', digest)))
        return result

    def setup(self, request, context):
        specs = pickle.loads(request.data)
        if Path(specs.pretrained_name_or_path).resolve() != MODEL or specs.policy_type != 'smolvla':
            raise ValueError('This server serves only the configured real SmolVLA checkpoint')
        if getattr(self, '_real_setup', None) == request.data and self.policy is not None:
            result = policy_server.services_pb2.Empty()
        else:
            result = original_setup(self, request, context)
            self._real_setup = bytes(request.data)
        self.policy.eval()
        self.policy.reset()
        for processor in (self.preprocessor, self.postprocessor):
            if hasattr(processor, 'reset'):
                processor.reset()
        context.set_trailing_metadata((('real-model-sha256', digest),))
        return result

    policy_server.PolicyServer.Ready = ready
    policy_server.PolicyServer.SendPolicyInstructions = setup


def decode_camera(value):
    if isinstance(value, dict):
        if value.get('encoding') != 'png_rgb_v1' or not isinstance(value.get('data'), bytes):
            raise ValueError('Unsupported camera encoding')
        bgr = cv2.imdecode(np.frombuffer(value['data'], dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError('Invalid camera PNG')
        value = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8 or value.shape != (480, 640, 3):
        raise ValueError('Expected native 640x480 RGB; download the updated run_real.py')
    return value


def training_observation(raw, features, policy_images):
    from lerobot.async_inference.helpers import raw_observation_to_observation
    native = dict(raw)
    native_features = {}
    for key, feature in policy_images.items():
        camera = key.removeprefix('observation.images.')
        frame = decode_camera(native[camera])
        native[camera] = frame
        native_features[key] = replace(feature, shape=(3, frame.shape[0], frame.shape[1]))
    # Same-size uint8 interpolation is identity; float conversion remains /255.
    # SmolVLA.prepare_images will do its original aspect-preserving resize/pad.
    return raw_observation_to_observation(native, features, native_features)


def main():
    load_settings(required=True)
    from lerobot.async_inference import policy_server
    policy_server.raw_observation_to_observation = training_observation
    install_server_protocol(policy_server)
    print('REAL IMAGE PIPELINE: native 640x480 RGB -> SmolVLA training resize/pad; no square warp.', flush=True)
    policy_server.serve()


if __name__ == '__main__':
    main()
