import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def linearize_image(img_pil):
    """Convert a PIL RGB image (sRGB) to linear RGB in [0,1]."""
    img_srgb = np.array(img_pil.convert("RGB"), dtype=np.float32) / 255.0
    mask = img_srgb <= 0.04045
    img_lin = np.where(mask, img_srgb / 12.92, ((img_srgb + 0.055) / 1.055) ** 2.4)
    return np.clip(img_lin, 0, 1)


def luminance(img_lin):
    """Compute human-eye weighted luminance (ITU-R BT.709) from linear RGB."""
    Y = 0.2126 * img_lin[..., 0] + 0.7152 * img_lin[..., 1] + 0.0722 * img_lin[..., 2]
    return np.clip(Y, 0, 1)


def estimate_illumination(Y, sigma_shade=100, method="linear"):
    """
    Estimate local illumination from a luminance map Y.
    Rescale so that the maximum Y corresponds to illumination = 1,
    ensuring that apparent albedo Y / illum_est ≤ 1.

    Parameters:
        Y: ndarray, luminance map in [0,1]
        sigma_shade: float, Gaussian smoothing parameter
        method: str, 'linear' or 'retinex'

    Returns:
        illum_est: ndarray, estimated illumination map
    """
    if method == "linear":
        illum_est = gaussian_filter(Y, sigma=sigma_shade)
    elif method == "retinex":
        logY = np.log(Y + 1e-6)  # avoid log(0)
        log_illum = gaussian_filter(logY, sigma=sigma_shade)
        illum_est = np.exp(log_illum)
    elif method == "mixed":
        raise NotImplementedError("might be nice to add in the future")
    else:
        raise ValueError("method must be 'linear' or 'retinex'")

    # Rescale so that brightest Y maps to illum = 1
    # TODO: maybe scale based on high percentile, mask out very bright or dark areas, or leave out alltogether?
    # max_Y = Y.max()
    # max_illum = illum_est.max()
    # if max_illum > 0:
    #     illum_est *= max_Y / max_illum

    return illum_est


def lightness_map(img_pil):
    """Compute simple lightness (HSL L) from a PIL RGB image."""
    return linearize_image(img_pil).mean(axis=-1)


def value_map(img_pil):
    """Compute simple brightness (HSV V) from a PIL RGB image."""
    return linearize_image(img_pil).max(axis=-1)


def luminance_map(img_pil):
    """Compute human-eye weighted luminance (ITU-R BT.709) from a PIL RGB image."""
    return luminance(linearize_image(img_pil))
