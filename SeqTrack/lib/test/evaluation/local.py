from lib.test.evaluation.environment import EnvSettings
import os

def local_env_settings():
    settings = EnvSettings()

    # === Base project path ===
    base = r"D:\Assignment_3\SeqTrack"

    # === Data paths ===
    settings.davis_dir = ''
    settings.got10k_lmdb_path = os.path.join(base, "data", "got10k_lmdb")
    settings.got10k_path = os.path.join(base, "data", "got10k")
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = os.path.join(base, "data", "lasot_extension_subset")
    settings.lasot_lmdb_path = os.path.join(base, "data", "lasot_lmdb")
    settings.lasot_path = os.path.join(base, "data", "lasot")
    settings.network_path = os.path.join(base, "test", "networks")   # Where tracking networks are stored
    settings.nfs_path = os.path.join(base, "data", "nfs")
    settings.otb_path = os.path.join(base, "data", "OTB2015")
    settings.prj_dir = base
    settings.result_plot_path = os.path.join(base, "test", "result_plots")
    settings.results_path = os.path.join(base, "test", "tracking_results")  # Where to store tracking results
    settings.save_dir = base
    settings.segmentation_path = os.path.join(base, "test", "segmentation_results")
    settings.tc128_path = os.path.join(base, "data", "TC128")
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = os.path.join(base, "data", "tnl2k")
    settings.tpl_path = ''
    settings.trackingnet_path = os.path.join(base, "data", "trackingnet")
    settings.uav_path = os.path.join(base, "data", "UAV123")
    settings.vot_path = os.path.join(base, "data", "VOT2019")
    settings.youtubevos_dir = ''

    return settings
