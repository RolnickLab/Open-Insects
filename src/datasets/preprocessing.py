# license removed

import calendar
import math

from functools import partial
import torch
import numpy as np
from torchvision import transforms
from dateutil.parser import parse as parse_date
from dateutil.parser import ParserError

import torchvision.transforms.functional as F


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, "constant")


# FLAGS = flags.FLAGS

# flags.DEFINE_enum(
#     "dataaug",
#     default="randaug",
#     enum_values=["randaug", "simple"],
#     help=("Data augmentation operations set"),
# )

# flags.DEFINE_bool(
#     "use_mixres", default=False, help=("Randomly apply downscaling to input image")
# )

# flags.DEFINE_integer(
#     "rescale_size",
#     default=None,
#     help=("Rescale image to size before any preprocessing step"),
# )


def random_resize(image, full_size=300):
    random_num = np.random.uniform()
    if random_num <= 0.25:
        transform = transforms.Resize((int(0.5 * full_size), int(0.5 * full_size)))
        image = transform(image)
    elif random_num <= 0.5:
        transform = transforms.Resize((int(0.25 * full_size), int(0.25 * full_size)))
        image = transform(image)

    return image


def get_image_transforms(config, input_size=224, is_training=True, preprocess_mode="torch"):
    if preprocess_mode == "torch":
        # imagenet preprocessing
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif preprocess_mode == "tf":
        # global butterfly preprocessing (-1 to 1)
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    else:
        # (0 to 1)
        mean, std = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]

    ops = []

    # ops += [SquarePad()]  # add padding

    if is_training:

        ops += [
            transforms.RandomResizedCrop(input_size, scale=(0.3, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
        ]
    
    else:
        ops += [transforms.Resize((input_size, input_size))]

    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]

    return transforms.Compose(ops)


def date2float(date):
    dt = parse_date(date).timetuple()
    year_days = 366 if calendar.isleap(dt.tm_year) else 365

    return dt.tm_yday / year_days


def encode_feat(feat, encode, concat_dim=0):
    if encode == "encode_cos_sin":
        return torch.cat((torch.sin(math.pi * feat), torch.cos(math.pi * feat)), concat_dim)
    else:
        raise RuntimeError("%s not implemented" % encode)

    return feat


def validate_loc_date(lat, lon, date):
    if lat is None or math.isnan(lat) or lat < -90.0 or lat > 90.0:
        return False
    if lon is None or math.isnan(lon) or lon < -180.0 or lon > 180.0:
        return False

    if date is None or not isinstance(date, str):
        return False
    else:
        try:
            parse_date(date)
        except ParserError:
            return False

    return True


def preprocess_loc_date(
    lat: float,
    lon: float,
    date: str,
    validate=True,
    loc_encode="encode_cos_sin",
    date_encode="encode_cos_sin",
):
    if validate:
        valid = validate_loc_date(lat, lon, date)
    else:
        valid = False

    if valid:
        lat = lat / 90.0
        lon = lon / 180.0
        date_c = date2float(date)
    else:
        lat = 0.0
        lon = 0.0
        date_c = 0.5

    lat = torch.tensor(lat).unsqueeze(-1)
    lat = encode_feat(lat, loc_encode)
    lon = torch.tensor(lon).unsqueeze(-1)
    lon = encode_feat(lon, loc_encode)
    feats = torch.cat((lat, lon), dim=0)

    date_c = date_c * 2.0 - 1.0
    date_c = torch.tensor(date_c).unsqueeze(-1)
    date_c = encode_feat(date_c, date_encode)
    feats = torch.cat((feats, date_c), dim=0)

    return feats.float(), torch.tensor(valid).float()
