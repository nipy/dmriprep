# -*- coding: utf-8 -*-

"""Console script for dmriprep."""
import os
import sys
import warnings

import click

from . import io
from . import run
from .data import get_dataset

# Filter warnings that are visible whenever you import another package that
# was compiled against an older numpy than is installed.
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


@click.command()
@click.option('--participant_label',
              help="The label(s) of the participant(s) that should be"
                   "analyzed. The label corresponds to"
                   "sub-<participant_label> from the BIDS spec (so it does"
                   "not include 'sub-'). If this parameter is not provided"
                   "all subjects will be analyzed. Multiple participants"
                   "can be specified with a space separated list.",
              default=None)
@click.option('--session',
              help="The session number off the participant data in the case of multiple sessions.",
              default=1)
@click.option('--denoise_strategy',
              help="Denoising strategy. Options are: mppca, localpca, or nlmeans.",
              default="mppca")
@click.option('--slice_outlier_threshold',
              help="Number of allowed outlier slices per volume. "
                   "If this is exceeded the volume is dropped from analysis. "
                   "If an int is provided, it is treated as number of allowed "
                   "outlier slices. If a float between 0 and 1 "
                   "(exclusive) is provided, it is treated the fraction of "
                   "allowed outlier slices.",
              default=0.05)
@click.argument('bids_dir',
                )
@click.argument('output_dir',
                )
@click.option('--plugin',
              help="Scheduling plugin type. Default is MultiProc.",
              default="MultiProc")
@click.option('--vox',
              help="Voxel resolution. Default is 2mm.",
              default='2mm')
@click.option('--verbose',
              help="Verbose logging. Default is False.",
              default=False)
@click.argument('analysis_level',
                type=click.Choice(['participant', 'group']),
                default='participant')
def main(participant_label, session, bids_dir, output_dir, plugin_type='MultiProc', verbose=False,
         denoise_strategy="mppca", slice_outlier_threshold=0.05, vox_size='2mm', analysis_level="participant"):
    """
    BIDS_DIR: The directory with the input dataset formatted according to
    the BIDS standard.

    OUTPUT_DIR: The directory where the output files should be stored.
    If you are running a group level analysis, this folder
    should be prepopulated with the results of
    the participant level analysis.

    ANALYSIS_LEVEL: Level of the analysis that will be performed. Multiple
    participant level analyses can be run independently
    (in parallel).
    """
    if analysis_level is not 'participant':
        raise NotImplementedError('The only valid analysis level for dmriprep '
                                  'is participant at the moment.')

    from dmriprep.io import get_bids_layout

    sub_dict = get_bids_layout(bids_dir, participant_label, session)
    dwi = sub_dict[session]['dwi']
    fbvec = sub_dict[session]['bvec']
    fbval = sub_dict[session]['bval']
    metadata = sub_dict[session]['metadata']

    wf = run.init_single_subject_wf(dwi, fbvec, fbval, metadata, out_dir=output_dir, strategy=denoise_strategy,
                                    vox_size=vox_size, plugin_type=plugin_type, outlier_thresh=slice_outlier_threshold,
                                    verbose=verbose)
    wf.run()

    return 0


@click.command()
@click.argument('output_dir')
@click.option('--subject', help="subject id to download (will choose 1 subject if not specified",
              default="sub-NDARBA507GCT")
@click.option('--study', help="which study to download. Right now we only support the HBN dataset",
              default="HBN")
def data(output_dir, study="HBN", subject="sub-NDARBA507GCT"):
    """
    Download dwi raw data in BIDS format from public datasets

    :param output_dir: A directory to write files to
    :param study: A study name, right now we only support 'HBN'
    :param subject: A subject from the study, starting with 'sub-'
    :return: None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if study.upper() != 'HBN':
        raise NotImplementedError('We only support data downloads from the HBN dataset right now.')

    get_dataset(os.path.abspath(output_dir), source=study.upper(), subject_id=subject)
    print('done')


@click.command()
@click.argument('output_dir')
@click.argument('bucket')
@click.option('--access_key', help="your AWS access key")
@click.option('--secret_key', help="your AWS access secret")
@click.option('--provider', default='s3', help="Cloud storage provider. Only S3 is supported right now.")
@click.option('--subject', default=None, help="Subject id to upload (optional)")
def upload(output_dir, bucket, access_key, secret_key, provider='s3', subject=None):
    """
    OUTPUT_DIR: The directory where the output files were stored.

    BUCKET: The cloud bucket name to upload data to.
    """
    import boto3
    from dask import compute, delayed
    from glob import glob
    from tqdm.auto import tqdm

    output_dir = os.path.abspath(output_dir)
    if not output_dir.endswith('/'):
        output_dir += '/'

    if provider == 's3' or provider == 'S3':
        client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)

        if subject is not None:
            assert os.path.exists(os.path.join(output_dir, subject)), 'this subject id does not exist!'
            subjects = [subject]
        else:
            subjects = [os.path.split(s)[1] for s in glob(os.path.join(output_dir, 'sub-*'))]

        def upload_subject(sub, sub_idx):
            base_dir = os.path.join(output_dir, sub, 'dmriprep')
            for root, dirs, files in os.walk(base_dir):
                if len(files):
                    for f in tqdm(files, desc=f"Uploading {sub} {root.split('/')[-1]}", position=sub_idx):
                        filepath = os.path.join(root, f)
                        key = root.replace(output_dir, '')
                        client.upload_file(filepath, bucket, os.path.join(key, f))

        uploads = [delayed(upload_subject)(s, idx) for idx, s in enumerate(subjects)]
        _ = list(compute(*uploads, scheduler="threads"))
    else:
        raise NotImplementedError('Only S3 is the only supported provider for data uploads at the moment')


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
