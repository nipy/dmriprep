import os

import nibabel as nib
import numpy as np


def make_gtab(fbval, fbvec, sesdir):
    from dipy.io import save_pickle
    from dipy.core.gradients import gradient_table
    from dipy.io import read_bvals_bvecs

    if fbval and fbvec:
        bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    else:
        raise ValueError("Either bvals or bvecs files not found (or rescaling failed)!")

    namer_dir = sesdir + "/dmri_tmp"
    if not os.path.isdir(namer_dir):
        os.mkdir(namer_dir)

    gtab_file = "%s%s" % (namer_dir, "/gtab.pkl")

    # Creating the gradient table
    gtab = gradient_table(bvals, bvecs)

    # Correct b0 threshold
    gtab.b0_threshold = min(bvals)

    # Show info
    print(gtab.info)

    # Save gradient table to pickle
    save_pickle(gtab_file, gtab)

    return gtab_file, gtab


def rescale_bvec(bvec, bvec_rescaled):
    """
    Normalizes b-vectors to be of unit length for the non-zero b-values. If the
    b-value is 0, the vector is untouched.

    Parameters
    ----------
    bvec : str
        File name of the original b-vectors file.
    bvec_rescaled : str
        File name of the new (normalized) b-vectors file. Must have extension `.bvec`.

    Returns
    -------
    bvec_rescaled : str
        File name of the new (normalized) b-vectors file. Must have extension `.bvec`.
    """
    bv1 = np.array(np.loadtxt(bvec))
    # Enforce proper dimensions
    bv1 = bv1.T if bv1.shape[0] == 3 else bv1

    # Normalize values not close to norm 1
    bv2 = [
        b / np.linalg.norm(b) if not np.isclose(np.linalg.norm(b), 0) else b
        for b in bv1
    ]
    np.savetxt(bvec_rescaled, bv2)
    return bvec_rescaled


def correct_vecs_and_make_b0s(fbval, fbvec, dwi_file, sesdir):
    from dipy.io import read_bvals_bvecs
    from dmriprep.core import make_gtab, rescale_bvec
    from dmriprep.utils import is_hemispherical

    namer_dir = sesdir + "/dmri_tmp"
    if not os.path.isdir(namer_dir):
        os.mkdir(namer_dir)

    bvec_rescaled = "%s%s" % (namer_dir, "/bvec_scaled.bvec")

    # loading bvecs/bvals
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    bvecs[np.where(np.any(abs(bvecs) >= 10, axis=1) == True)] = [1, 0, 0]
    bvecs[np.where(bvals == 0)] = 0
    if (
        len(
            bvecs[
                np.where(
                    np.logical_and(
                        bvals > 50, np.all(abs(bvecs) == np.array([0, 0, 0]), axis=1)
                    )
                )
            ]
        )
        > 0
    ):
        raise ValueError(
            "WARNING: Encountered potentially corrupted bval/bvecs. Check to ensure volumes with a "
            "diffusion weighting are not being treated as B0's along the bvecs"
        )
    np.savetxt(fbval, bvals)
    np.savetxt(fbvec, bvecs)
    bvec_rescaled = rescale_bvec(fbvec, bvec_rescaled)
    vecs_rescaled = np.genfromtxt(bvec_rescaled)
    vecs = np.round(vecs_rescaled, 8)[~(np.round(vecs_rescaled, 8) == 0).all(1)]
    [is_hemi, pole] = is_hemispherical(vecs)
    if is_hemi is True:
        raise ValueError(
            "B-vectors for this data are hemispherical and therefore use of topup/eddy routines is not "
            "advised"
        )

    [gtab_file, gtab] = make_gtab(fbval, bvec_rescaled, sesdir)

    # Get b0 indices
    b0s = np.where(gtab.bvals == gtab.b0_threshold)[0].tolist()
    print("%s%s" % ("b0's found at: ", b0s))
    firstb0 = b0s[0]

    # Extract and Combine all b0s collected
    print("Extracting b0's...")
    b0_vols = []
    dwi_img = nib.load(dwi_file)
    dwi_data = dwi_img.get_data()
    for b0 in b0s:
        print(b0)
        b0_vols.append(dwi_data[:, :, :, b0])

    firstb0_file = sesdir + "/firstb0.nii.gz"
    nib.save(
        nib.Nifti1Image(
            b0_vols[firstb0].astype(np.float32), dwi_img.affine, dwi_img.header
        ),
        firstb0_file,
    )

    return firstb0, firstb0_file, gtab_file, b0_vols, b0s


def topup_inputs_from_dwi_files(dwi, sesdir, spec_acqp, b0_vols, topup_config_odd):
    from collections import defaultdict

    """Create a datain spec and a slspec from a list of dwi files."""
    # Write the datain.txt file
    datain_lines = []
    spec_counts = defaultdict(int)

    dwi_img = nib.load(dwi)
    imain_data = []
    for b0_vol in b0_vols:
        num_trs = 1 if len(b0_vol.shape) < 4 else b0_vol.shape[3]
        datain_lines.extend([spec_acqp] * num_trs)
        spec_counts[spec_acqp] += num_trs
        imain_data.append(b0_vol)

    # Make a 4d series
    imain_output = sesdir + "/topup_imain.nii.gz"
    imain_data_4d = [imain_vol[..., np.newaxis] for imain_vol in imain_data]
    imain_img = nib.Nifti1Image(
        np.concatenate(imain_data_4d, 3), dwi_img.affine, dwi_img.header
    )
    assert imain_img.shape[3] == len(datain_lines)
    imain_img.to_filename(imain_output)

    # Write the datain text file
    datain_file = sesdir + "/acqparams.txt"
    with open(datain_file, "w") as f:
        f.write("\n".join(datain_lines))

    example_b0 = b0_vols[0]
    topup_config = "b02b0.cnf"
    if 1 in (example_b0.shape[0] % 2, example_b0.shape[1] % 2, example_b0.shape[2] % 2):
        print("Using slower b02b0_1.cnf because an axis has an odd number of slices")
        topup_config = topup_config_odd

    return datain_file, imain_output, example_b0, datain_lines, topup_config


def eddy_inputs_from_dwi_files(sesdir, gtab_file):
    from dipy.io import load_pickle

    b0s_mask_all = []
    gtab = load_pickle(gtab_file)
    b0s_mask = gtab.b0s_mask
    b0s_mask_all.append(b0s_mask)

    # Create the index file
    index_file = sesdir + "/index.txt"
    ix_vec = []
    i = 1
    pastfirst0s = False
    for val in gtab.bvals:
        if val > gtab.b0_threshold:
            pastfirst0s = True
        elif val <= gtab.b0_threshold and pastfirst0s is True:
            i = i + 1
        ix_vec.append(i)
    with open(index_file, "w") as f:
        f.write(" ".join(map(str, ix_vec)))

    return index_file


def id_outliers_fn(outlier_report, threshold, dwi_file):
    """Get list of scans that exceed threshold for number of outliers
    Parameters
    ----------
    outlier_report: string
        Path to the fsl_eddy outlier report
    threshold: int or float
        If threshold is an int, it is treated as number of allowed outlier
        slices. If threshold is a float between 0 and 1 (exclusive), it is
        treated the fraction of allowed outlier slices before we drop the
        whole volume.
    dwi_file: string
        Path to nii dwi file to determine total number of slices
    Returns
    -------
    drop_scans: numpy.ndarray
        List of scan indices to drop
    """
    import nibabel as nib
    import numpy as np
    import os.path as op
    import parse

    with open(op.abspath(outlier_report), "r") as fp:
        lines = fp.readlines()

    p = parse.compile(
        "Slice {slice:d} in scan {scan:d} is an outlier with "
        "mean {mean_sd:f} standard deviations off, and mean "
        "squared {mean_sq_sd:f} standard deviations off."
    )

    outliers = [p.parse(l).named for l in lines]
    scans = {d["scan"] for d in outliers}

    def num_outliers(scan, outliers):
        return len([d for d in outliers if d["scan"] == scan])

    if 0 < threshold < 1:
        img = nib.load(dwi_file)
        try:
            threshold *= img.header.get_n_slices()
        except nib.spatialimages.HeaderDataError:
            print(
                "WARNING. We are not sure which dimension has the "
                "slices in this image. So we are using the 3rd dim.",
                img.shape,
            )
            threshold *= img.shape[2]

    drop_scans = np.array([s for s in scans if num_outliers(s, outliers) > threshold])

    outpath = op.abspath("dropped_scans.txt")
    np.savetxt(outpath, drop_scans, fmt="%d")

    return drop_scans, outpath


def drop_outliers_fn(in_file, in_bval, in_bvec, drop_scans):
    """Drop outlier volumes from dwi file
    Parameters
    ----------
    in_file: string
        Path to nii dwi file to drop outlier volumes from
    in_bval: string
        Path to bval file to drop outlier volumes from
    in_bvec: string
        Path to bvec file to drop outlier volumes from
    drop_scans: numpy.ndarray or str
        List of scan indices to drop. If str, then assume path to text file
        listing outlier volumes.

    Returns
    -------
    out_file: string
        Path to "thinned" output dwi file
    out_bval: string
        Path to "thinned" output bval file
    out_bvec: string
        Path to "thinned" output bvec file
    """
    import nibabel as nib
    import numpy as np
    import os.path as op
    from nipype.utils.filemanip import fname_presuffix

    if isinstance(drop_scans, str):
        try:
            drop_scans = np.genfromtxt(drop_scans)
        except TypeError:
            print(
                "Unrecognized file format. Unable to load vector from drop_scans file."
            )

    img = nib.load(op.abspath(in_file))
    img_data = img.get_data()
    img_data_thinned = np.delete(img_data, drop_scans, axis=3)
    if isinstance(img, nib.nifti1.Nifti1Image):
        img_thinned = nib.Nifti1Image(
            img_data_thinned.astype(np.float64), img.affine, header=img.header
        )
    elif isinstance(img, nib.nifti2.Nifti2Image):
        img_thinned = nib.Nifti2Image(
            img_data_thinned.astype(np.float64), img.affine, header=img.header
        )
    else:
        raise TypeError("in_file does not contain Nifti image datatype.")

    out_file = fname_presuffix(in_file, suffix="_thinned", newpath=op.abspath("."))
    nib.save(img_thinned, op.abspath(out_file))

    bval = np.loadtxt(in_bval)
    bval_thinned = np.delete(bval, drop_scans, axis=0)
    out_bval = fname_presuffix(in_bval, suffix="_thinned", newpath=op.abspath("."))
    np.savetxt(out_bval, bval_thinned)

    bvec = np.loadtxt(in_bvec)
    bvec_thinned = np.delete(bvec, drop_scans, axis=0)
    out_bvec = fname_presuffix(in_bvec, suffix="_thinned", newpath=op.abspath("."))
    np.savetxt(out_bvec, bvec_thinned)

    print("%s%s" % ('Dropping outlier volumes:\n', drop_scans))
    return out_file, out_bval, out_bvec


def get_params(A):
    """This is a copy of spm's spm_imatrix where
    we already know the rotations and translations matrix,
    shears and zooms (as outputs from fsl FLIRT/avscale)
    Let A = the 4x4 rotation and translation matrix
    R = [          c5*c6,           c5*s6, s5]
        [-s4*s5*c6-c4*s6, -s4*s5*s6+c4*c6, s4*c5]
        [-c4*s5*c6+s4*s6, -c4*s5*s6-s4*c6, c4*c5]
    """

    def rang(b):
        a = min(max(b, -1), 1)
        return a

    Ry = np.arcsin(A[0, 2])
    # Rx = np.arcsin(A[1, 2] / np.cos(Ry))
    # Rz = np.arccos(A[0, 1] / np.sin(Ry))

    if (abs(Ry) - np.pi / 2) ** 2 < 1e-9:
        Rx = 0
        Rz = np.arctan2(-rang(A[1, 0]), rang(-A[2, 0] / A[0, 2]))
    else:
        c = np.cos(Ry)
        Rx = np.arctan2(rang(A[1, 2] / c), rang(A[2, 2] / c))
        Rz = np.arctan2(rang(A[0, 1] / c), rang(A[0, 0] / c))

    rotations = [Rx, Ry, Rz]
    translations = [A[0, 3], A[1, 3], A[2, 3]]

    return rotations, translations


def get_flirt_motion_parameters(flirt_mats):
    import os.path as op
    from nipype.interfaces.fsl.utils import AvScale
    from dmriprep.core import get_params

    motion_params = open(op.abspath("motion_parameters.par"), "w")
    for mat in flirt_mats:
        res = AvScale(mat_file=mat).run()
        A = np.asarray(res.outputs.rotation_translation_matrix)
        rotations, translations = get_params(A)
        for i in rotations + translations:
            motion_params.write("%f " % i)
        motion_params.write("\n")
    motion_params.close()
    motion_params = op.abspath("motion_parameters.par")
    return motion_params


def read_nifti_sidecar(json_file):
    import json

    with open(json_file, "r") as f:
        metadata = json.load(f)
    pe_dir = metadata["PhaseEncodingDirection"]
    slice_times = metadata.get("SliceTiming")
    trt = metadata.get("TotalReadoutTime")
    if trt is None:
        pass

    return {
        "PhaseEncodingDirection": pe_dir,
        "SliceTiming": slice_times,
        "TotalReadoutTime": trt,
    }


def extract_metadata(metadata):
    from dmriprep.core import read_nifti_sidecar

    acqp_lines = {
        "i": "1 0 0 %.6f",
        "j": "0 1 0 %.6f",
        "k": "0 0 1 %.6f",
        "i-": "-1 0 0 %.6f",
        "j-": "0 -1 0 %.6f",
        "k-": "0 0 -1 %.6f",
    }
    spec = read_nifti_sidecar(metadata)
    spec_line = acqp_lines[spec["PhaseEncodingDirection"]]
    spec_acqp = spec_line % spec["TotalReadoutTime"]
    spec_slice = spec["SliceTiming"]
    return spec_line, spec_acqp, spec_slice


def check_shelled(gtab_file):
    from dipy.io import load_pickle

    # Check whether data is shelled
    gtab = load_pickle(gtab_file)
    if len(np.unique(gtab.bvals)) > 2:
        is_shelled = True
    else:
        is_shelled = False
    return is_shelled


def make_mean_b0(in_file):
    import time

    b0_img = nib.load(in_file)
    b0_img_data = b0_img.get_data()
    mean_b0 = np.mean(b0_img_data, axis=3, dtype=b0_img_data.dtype)
    mean_file_out = in_file.split(".nii.gz")[0] + "_mean.nii.gz"
    nib.save(
        nib.Nifti1Image(mean_b0, affine=b0_img.affine, header=b0_img.header),
        mean_file_out,
    )
    while os.path.isfile(mean_file_out) is False:
        time.sleep(1)
    return mean_file_out


def suppress_gibbs(in_file, sesdir):
    from time import time
    from dipy.denoise.gibbs import gibbs_removal

    t = time()
    img = nib.load(in_file)
    img_data = img.get_data()
    gibbs_corr_data = gibbs_removal(img_data)
    print("Time taken for gibbs suppression: ", -t + time())
    gibbs_free_file = sesdir + "/gibbs_free_data.nii.gz"
    nib.save(
        nib.Nifti1Image(gibbs_corr_data.astype(np.float32), img.affine, img.header),
        gibbs_free_file,
    )
    return gibbs_corr_data, gibbs_free_file


def denoise(
    in_file,
    sesdir,
    gtab_file,
    mask,
    strategy,
    N=4,
    patch_radius=2,
    smooth_factor=3,
    tau_factor=2.3,
    block_radius=1,
):
    from time import time
    from dipy.denoise.noise_estimate import estimate_sigma
    from dipy.denoise.pca_noise_estimate import pca_noise_estimate
    from dipy.denoise.localpca import localpca, mppca
    from dipy.denoise.nlmeans import nlmeans
    from dipy.io import load_pickle

    gtab = load_pickle(gtab_file)
    t = time()
    img = nib.load(in_file)
    img_data = img.get_data()
    if strategy == "mppca":
        img_data_den = mppca(img_data, patch_radius=patch_radius)
    elif strategy == "localpca":
        sigma = pca_noise_estimate(
            img_data, gtab, correct_bias=True, smooth=smooth_factor
        )
        img_data_den = localpca(
            img_data, sigma, tau_factor=tau_factor, patch_radius=patch_radius
        )
    elif strategy == "nlmeans":
        sigma = estimate_sigma(img_data, N=N)
        img_data_den = nlmeans(
            img_data,
            sigma=sigma,
            mask=mask,
            patch_radius=patch_radius,
            block_radius=block_radius,
            rician=True,
        )
    else:
        raise ValueError("Denoising strategy not recognized.")
    print("Time taken for denoising: ", -t + time())
    denoised_file = sesdir + "/preprocessed_data_denoised_" + strategy + ".nii.gz"
    nib.save(
        nib.Nifti1Image(img_data_den.astype(np.float32), img.affine, img.header),
        denoised_file,
    )
    return img_data_den, denoised_file
