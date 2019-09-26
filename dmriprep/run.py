import os


def init_single_subject_wf(
    sub,
    ses,
    dwi,
    fbval,
    fbvec,
    metadata,
    out_dir,
    strategy="mppca",
    vox_size="2mm",
    plugin_type="MultiProc",
    outlier_thresh=0.10,
    verbose=False
):
    import json
    from pathlib import Path
    from nipype.pipeline import engine as pe
    from nipype.interfaces import utility as niu
    from nipype.interfaces import fsl
    from nipype.algorithms.rapidart import ArtifactDetect
    from dmriprep import interfaces, core, qc

    import_list = [
        "import sys",
        "import os",
        "import numpy as np",
        "import nibabel as nib",
        "import warnings",
        'warnings.filterwarnings("ignore")',
    ]

    subdir = "%s%s%s" % (out_dir, "/", sub)
    if not os.path.isdir(subdir):
        os.mkdir(subdir)
    sesdir = "%s%s%s%s%s" % (out_dir, "/", sub, "/ses-", ses)
    if not os.path.isdir(sesdir):
        os.mkdir(sesdir)

    eddy_cfg_file = "%s%s" % (str(Path(__file__).parent), "/eddy_params.json")
    topup_config_odd = "%s%s" % (str(Path(__file__).parent), "/b02b0_1.cnf")

    # Create dictionary of eddy args
    with open(eddy_cfg_file, "r") as f:
        eddy_args = json.load(f)

    wf = pe.Workflow(name="single_subject_dmri")
    wf.base_dir = sesdir
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "dwi",
                "fbvec",
                "fbval",
                "metadata",
                "sub",
                "ses",
                "sesdir",
                "strategy",
                "vox_size",
                "outlier_thresh",
                "eddy_cfg_file",
                "topup_config_odd",
            ]
        ),
        name="inputnode",
    )
    inputnode.inputs.dwi = dwi
    inputnode.inputs.fbvec = fbvec
    inputnode.inputs.fbval = fbval
    inputnode.inputs.metadata = metadata
    inputnode.inputs.sub = sub
    inputnode.inputs.ses = ses
    inputnode.inputs.sesdir = sesdir
    inputnode.inputs.strategy = strategy
    inputnode.inputs.vox_size = vox_size
    inputnode.inputs.outlier_thresh = outlier_thresh
    inputnode.inputs.eddy_cfg_file = eddy_cfg_file
    inputnode.inputs.topup_config_odd = topup_config_odd

    check_orient_and_dims_dwi_node = pe.Node(
        niu.Function(
            input_names=["infile", "vox_size", "bvecs", "outdir"],
            output_names=["outfile", "bvecs"],
            function=qc.check_orient_and_dims,
            imports=import_list,
        ),
        name="check_orient_and_dims_dwi_node",
    )

    # Make gtab and get b0 indices
    correct_vecs_and_make_b0s_node = pe.Node(
        niu.Function(
            input_names=["fbval", "fbvec", "dwi_file", "sesdir"],
            output_names=["firstb0", "firstb0_file", "gtab_file", "b0_vols", "b0s"],
            function=core.correct_vecs_and_make_b0s,
            imports=import_list,
        ),
        name="correct_vecs_and_make_b0s",
    )

    btr_node_premoco = pe.Node(fsl.BET(), name="bet_pre_moco")
    btr_node_premoco.inputs.mask = True
    btr_node_premoco.inputs.frac = 0.2

    apply_mask_premoco_node = pe.Node(fsl.ApplyMask(), name="apply_mask_pre_moco")

    # Detect and remove motion outliers
    fsl_split_node = pe.Node(fsl.Split(dimension="t"), name="fsl_split")

    pick_ref = pe.Node(niu.Select(), name="pick_ref")

    coreg = pe.MapNode(
        fsl.FLIRT(no_search=True, interp="spline", padding_size=1, dof=6),
        name="coregistration",
        iterfield=["in_file"],
    )

    get_motion_params_node = pe.Node(
        niu.Function(
            input_names=["flirt_mats"],
            output_names=["motion_params"],
            function=core.get_flirt_motion_parameters,
            imports=import_list,
        ),
        name="get_motion_params",
    )

    fsl_merge_node = pe.Node(fsl.Merge(dimension="t"), name="fsl_merge")

    art_node = pe.Node(interface=ArtifactDetect(), name="art")
    art_node.inputs.use_differences = [True, True]
    art_node.inputs.save_plot = False
    art_node.inputs.use_norm = True
    art_node.inputs.norm_threshold = 3
    art_node.inputs.zintensity_threshold = 9
    art_node.inputs.mask_type = "spm_global"
    art_node.inputs.parameter_source = "FSL"

    drop_outliers_fn_node = pe.Node(
        niu.Function(
            input_names=["in_file", "in_bval", "in_bvec", "drop_scans"],
            output_names=["out_file", "out_bval", "out_bvec"],
            function=core.drop_outliers_fn,
            imports=import_list,
        ),
        name="drop_outliers_fn",
    )

    make_gtab_node = pe.Node(
        niu.Function(
            input_names=["fbval", "fbvec", "sesdir"],
            output_names=["gtab_file", "gtab"],
            function=core.make_gtab,
            imports=import_list,
        ),
        name="make_gtab",
    )

    extract_metadata_node = pe.Node(
        niu.Function(
            input_names=["metadata"],
            output_names=["spec_line", "spec_acqp", "spec_slice"],
            function=core.extract_metadata,
            imports=import_list,
        ),
        name="extract_metadata",
    )

    # Gather TOPUP/EDDY inputs
    check_shelled_node = pe.Node(
        niu.Function(
            input_names=["gtab_file"],
            output_names=["check_shelled"],
            function=core.check_shelled,
            imports=import_list,
        ),
        name="check_shelled",
    )

    get_topup_inputs_node = pe.Node(
        niu.Function(
            input_names=["dwi", "sesdir", "spec_acqp", "b0_vols", "topup_config_odd"],
            output_names=[
                "datain_file",
                "imain_output",
                "example_b0",
                "datain_lines",
                "topup_config",
            ],
            function=core.topup_inputs_from_dwi_files,
            imports=import_list,
        ),
        name="get_topup_inputs",
    )

    get_eddy_inputs_node = pe.Node(
        niu.Function(
            input_names=["sesdir", "gtab_file"],
            output_names=["index_file"],
            function=core.eddy_inputs_from_dwi_files,
            imports=import_list,
        ),
        name="get_eddy_inputs",
    )

    # Run TOPUP
    topup_node = pe.Node(fsl.TOPUP(), name="topup")
    topup_node._mem_gb = 4
    topup_node.n_procs = 1
    topup_node.interface.mem_gb = 4
    topup_node.interface.n_procs = 1

    # Run BET on mean b0 of topup-corrected output
    make_mean_b0_node = pe.Node(
        niu.Function(
            input_names=["in_file"],
            output_names=["mean_file_out"],
            function=core.make_mean_b0,
            imports=import_list,
        ),
        name="make_mean_b0",
    )
    btr_node = pe.Node(fsl.BET(), name="bet")
    btr_node.inputs.mask = True
    btr_node.inputs.frac = 0.2

    # Run EDDY
    eddy_node = pe.Node(interfaces.ExtendedEddy(**eddy_args), name="eddy")
    eddy_node.inputs.num_threads = 4
    eddy_node._mem_gb = 8
    eddy_node.n_procs = 4
    eddy_node.interface.mem_gb = 8
    eddy_node.interface.n_procs = 4

    # Drop outlier volumes
    id_outliers_fn_node = pe.Node(
        niu.Function(
            input_names=["outlier_report", "threshold", "dwi_file"],
            output_names=["drop_scans", "outpath"],
            function=core.id_outliers_fn,
            imports=import_list,
        ),
        name="id_outliers_fn",
    )

    drop_outliers_fn_posteddy_node = pe.Node(
        niu.Function(
            input_names=["in_file", "in_bval", "in_bvec", "drop_scans"],
            output_names=["out_file", "out_bval", "out_bvec"],
            function=core.drop_outliers_fn,
            imports=import_list,
        ),
        name="drop_outliers_fn_posteddy",
    )

    make_gtab_node_final = pe.Node(
        niu.Function(
            input_names=["fbval", "fbvec", "sesdir"],
            output_names=["gtab_file", "gtab"],
            function=core.make_gtab,
            imports=import_list,
        ),
        name="make_gtab_final",
    )

    apply_mask_node = pe.Node(fsl.ApplyMask(), name="apply_mask")

    # Suppress gibbs ringing and denoise
    suppress_gibbs_node = pe.Node(
        niu.Function(
            input_names=["in_file", "sesdir"],
            output_names=["gibbs_corr_data", "gibbs_free_file"],
            function=core.suppress_gibbs,
            imports=import_list,
        ),
        name="suppress_gibbs",
    )
    suppress_gibbs_node._mem_gb = 2
    suppress_gibbs_node.n_procs = 1

    denoise_node = pe.Node(
        niu.Function(
            input_names=["in_file", "sesdir", "gtab_file", "mask", "strategy"],
            output_names=["img_data_den", "denoised_file"],
            function=core.denoise,
            imports=import_list,
        ),
        name="denoise",
    )
    denoise_node._mem_gb = 4
    denoise_node.n_procs = 1

    outputnode = pe.Node(
        niu.IdentityInterface(fields=["preprocessed_data", "final_bvec", "final_bval"]),
        name="outputnode",
    )

    wf.connect([
                (inputnode, check_orient_and_dims_dwi_node, [("fbvec", "bvecs"),
                                                             ("dwi", "infile"),
                                                             ("vox_size", "vox_size"),
                                                             ("sesdir", "outdir")]),
                (inputnode, correct_vecs_and_make_b0s_node, [("fbval", "fbval"),
                                                             ("sesdir", "sesdir")]),
                (check_orient_and_dims_dwi_node, correct_vecs_and_make_b0s_node, [("bvecs", "fbvec"),
                                                                                  ("outfile", "dwi_file")]),
                (correct_vecs_and_make_b0s_node, btr_node_premoco, [("firstb0_file", "in_file")]),
                (btr_node_premoco, apply_mask_premoco_node, [("mask_file", "mask_file")]),
                (check_orient_and_dims_dwi_node, apply_mask_premoco_node, [("outfile", "in_file")]),
                (apply_mask_premoco_node, fsl_split_node, [("out_file", "in_file")]),
                (correct_vecs_and_make_b0s_node, pick_ref, [("firstb0", "index")]),
                (fsl_split_node, pick_ref, [("out_files", "inlist")]),
                (pick_ref, coreg, [("out", "reference")]),
                (fsl_split_node, coreg, [("out_files", "in_file")]),
                (coreg, get_motion_params_node, [("out_matrix_file", "flirt_mats")]),
                (coreg, fsl_merge_node, [("out_file", "in_files")]),
                (get_motion_params_node, art_node, [("motion_params", "realignment_parameters")]),
                (fsl_merge_node, art_node, [("merged_file", "realigned_files")]),
                (check_orient_and_dims_dwi_node, drop_outliers_fn_node, [("bvecs", "in_bvec"),
                                                                         ("outfile", "in_file")]),
                (inputnode, drop_outliers_fn_node, [("fbval", "in_bval")]),
                (art_node, drop_outliers_fn_node, [("outlier_files", "drop_scans")]),
                (inputnode, extract_metadata_node, [("metadata", "metadata")]),
                (correct_vecs_and_make_b0s_node, check_shelled_node, [("gtab_file", "gtab_file")]),
                (drop_outliers_fn_node, get_topup_inputs_node, [("out_file", "dwi")]),
                (inputnode, get_topup_inputs_node, [("sesdir", "sesdir"),
                                                    ("topup_config_odd", "topup_config_odd")]),
                (correct_vecs_and_make_b0s_node, get_topup_inputs_node, [("b0_vols", "b0_vols")]),
                (extract_metadata_node, get_topup_inputs_node, [("spec_acqp", "spec_acqp")]),
                (inputnode, get_eddy_inputs_node, [("sesdir", "sesdir")]),
                (drop_outliers_fn_node, make_gtab_node, [("out_bvec", "fbvec"),
                                                         ("out_bval", "fbval")]),
                (inputnode, make_gtab_node, [("sesdir", "sesdir")]),
                (make_gtab_node, get_eddy_inputs_node, [("gtab_file", "gtab_file")]),
                (extract_metadata_node, get_eddy_inputs_node, [("spec_acqp", "spec_acqp")]),
                (get_topup_inputs_node, topup_node, [("datain_file", "encoding_file"),
                                                     ("imain_output", "in_file"),
                                                     ("topup_config", "config")]),
                (topup_node, make_mean_b0_node, [("out_corrected", "in_file")]),
                (make_mean_b0_node, btr_node, [("mean_file_out", "in_file")]),
                (check_shelled_node, eddy_node, [("check_shelled", "is_shelled")]),
                (btr_node, eddy_node, [("mask_file", "in_mask")]),
                (get_eddy_inputs_node, eddy_node, [("index_file", "in_index")]),
                (get_topup_inputs_node, eddy_node, [("datain_file", "in_acqp")]),
                (topup_node, eddy_node, [("out_movpar", "in_topup_movpar"),
                                         ("out_fieldcoef", "in_topup_fieldcoef")]),
                (drop_outliers_fn_node, eddy_node, [("out_file", "in_file"),
                                                    ("out_bval", "in_bval"),
                                                    ("out_bvec", "in_bvec")]),
                (eddy_node, id_outliers_fn_node, [("out_outlier_report", "outlier_report"),
                                                  ("out_corrected", "dwi_file")]),
                (inputnode, id_outliers_fn_node, [("outlier_thresh", "threshold")]),
                (drop_outliers_fn_node, drop_outliers_fn_posteddy_node, [("out_bval", "in_bval")]),
                (eddy_node, drop_outliers_fn_posteddy_node, [("out_corrected", "in_file"),
                                                             ("out_rotated_bvecs", "in_bvec")]),
                (id_outliers_fn_node, drop_outliers_fn_posteddy_node, [("drop_scans", "drop_scans")]),
                (drop_outliers_fn_posteddy_node, apply_mask_node, [("out_file", "in_file")]),
                (btr_node, apply_mask_node, [("mask_file", "mask_file")]),
                (apply_mask_node, suppress_gibbs_node, [("out_file", "in_file")]),
                (inputnode, denoise_node, [("strategy", "strategy")]),
                (btr_node, denoise_node, [("mask_file", "mask")]),
                (drop_outliers_fn_posteddy_node, make_gtab_node_final, [("out_bvec", "fbvec"),
                                                                        ("out_bval", "fbval")]),
                (inputnode, make_gtab_node_final, [("sesdir", "sesdir")]),
                (make_gtab_node_final, denoise_node, [("gtab_file", "gtab_file")]),
                (suppress_gibbs_node, denoise_node, [("gibbs_free_file", "in_file")]),
                (inputnode, suppress_gibbs_node, [("sesdir", "sesdir")]),
                (inputnode, denoise_node, [("sesdir", "sesdir")]),
                (denoise_node, outputnode, [("denoised_file", "preprocessed_data")]),
                (drop_outliers_fn_posteddy_node, outputnode, [("out_bvec", "final_bvec"),
                                                              ("out_bval", "final_bval")])
                ])

    if verbose is True:
        from nipype import config, logging
        cfg_v = dict(logging={'workflow_level': 'DEBUG', 'utils_level': 'DEBUG', 'interface_level': 'DEBUG',
                              'log_directory': str(wf.base_dir), 'log_to_file': True},
                     monitoring={'enabled': True, 'sample_frequency': '0.1', 'summary_append': True,
                                 'summary_file': str(wf.base_dir)})
        logging.update_logging(config)
        config.update_config(cfg_v)
        config.enable_debug_mode()
        config.enable_resource_monitor()

        import logging
        callback_log_path = "%s%s" % (wf.base_dir, '/run_stats.log')
        logger = logging.getLogger('callback')
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(callback_log_path)
        logger.addHandler(handler)

    # Set runtime/logging configurations
    cfg = dict(
        execution={
            "stop_on_first_crash": True,
            "hash_method": "content",
            "crashfile_format": "txt",
            "display_variable": ":0",
            "job_finished_timeout": 65,
            "matplotlib_backend": "Agg",
            "plugin": str(plugin_type),
            "use_relative_paths": True,
            "parameterize_dirs": True,
            "remove_unnecessary_outputs": False,
            "remove_node_directories": False,
            "raise_insufficient": True,
            "poll_sleep_duration": 0.01,
        }
    )
    for key in cfg.keys():
        for setting, value in cfg[key].items():
            wf.config[key][setting] = value

    try:
        wf.write_graph(graph2use="colored", format='png')
    except:
        pass

    return wf
