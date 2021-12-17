import copy
import io
import logging
import urllib.parse
import warnings

from google.cloud import storage as gstorage
import torch
from theconf import Config as C


formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
warnings.filterwarnings(
    "ignore", "DeprecationWarning: 'saved_variables' is deprecated", UserWarning
)


def get_logger(name, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


logger = get_logger(__name__, level=logging.INFO)


def add_filehandler(logger, filepath, level=logging.DEBUG):
    fh = logging.FileHandler(filepath)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


class EMA:
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}

    def state_dict(self):
        return copy.deepcopy(self.shadow)

    def __len__(self):
        return len(self.shadow)

    def __call__(self, module, step=None):
        if step is None:
            mu = self.mu
        else:
            # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/train/ExponentialMovingAverage?hl=PL
            mu = min(self.mu, (1.0 + step) / (10 + step))

        for name, x in module.state_dict().items():
            if name in self.shadow:
                new_average = (1.0 - mu) * x + mu * self.shadow[name]
                self.shadow[name] = new_average.clone()
            else:
                self.shadow[name] = x.clone()


def urlunparse(path, cloud, bucket):
    return urllib.parse.urlunparse([cloud, bucket, path, "", "", ""])


def parse_path(dirname):
    """
    * dirname can be
      - a local directory (full or relative): /data/models/cifar10
      - google storage path: "gs://bucketname/foo/bar"
    * Returns (scheme, bucket, path)
      - local dir: "", "", "/data/models/cifar10"  (scheme and bucket are empty)
      - gs: "gs", "bucketname", "foo/bar"
    * If the scheme is "gs", the initial "/" is stripped
    """
    p = urllib.parse.urlparse(dirname)
    path = p.path[1:] if p.scheme == "gs" else p.path
    assert path, "Empty path"
    return p.scheme, p.netloc, path


def model_load(fullpath: str):
    cloud, bucket, path = parse_path(fullpath)
    if not cloud:
        return torch.load(path)
    client = gstorage.Client.from_service_account_info(C.get()["GSCREDS"])
    bucket = client.get_bucket(bucket)
    blob = bucket.get_blob(path)
    if not blob:
        raise FileNotFoundError(f"path does not exist: {fullpath}")
    blob = blob.download_as_bytes()
    buffer = io.BytesIO(blob)
    buffer.seek(0)
    return torch.load(buffer)


def model_save(obj, fullpath: str):
    cloud, bucket, path = parse_path(fullpath)
    if not cloud:
        return torch.save(obj, path)
    client = gstorage.Client.from_service_account_info(C.get()["GSCREDS"])
    bucket = client.get_bucket(bucket)
    blob = bucket.blob(path)
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    buffer.seek(0)
    blob.upload_from_file(buffer)
    logger.info("blob: %s", blob)


def test_creds():
    print(C.get()["GSCREDS"])
