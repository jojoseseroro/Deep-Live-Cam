def paste_back_region(swapped_patch, orig_patch, region_mask, preserve_strength=1.0):
    """Paste back original patch into swapped_patch guided by region_mask.

    swapped_patch: HxWx3 uint8
    orig_patch: HxWx3 uint8
    region_mask: HxW float32 in [0,1] describing the region where origin should be
                 preserved (1.0 means full preservation of orig_patch)
    preserve_strength: float in [0,1], how strongly to preserve original

    Returns a new swapped_patch with orig areas pasted/blended according to mask.
    """
    # Ensure shapes
    import numpy as np
    import cv2

    if swapped_patch is None or orig_patch is None:
        return swapped_patch
    try:
        if swapped_patch.shape != orig_patch.shape:
            orig_rs = cv2.resize(orig_patch, (swapped_patch.shape[1], swapped_patch.shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            orig_rs = orig_patch

        # region_mask should match patch size
        if region_mask.shape != swapped_patch.shape[:2]:
            mask_rs = cv2.resize(region_mask, (swapped_patch.shape[1], swapped_patch.shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            mask_rs = region_mask

        # compute final preservation alpha in [0,1]
        preserve_alpha = np.clip(mask_rs.astype('float32') * float(preserve_strength), 0.0, 1.0)
        alpha3 = np.expand_dims(preserve_alpha, axis=2)

        blended = (orig_rs.astype('float32') * alpha3 + swapped_patch.astype('float32') * (1.0 - alpha3)).astype('uint8')
        return blended
    except Exception:
        return swapped_patch
