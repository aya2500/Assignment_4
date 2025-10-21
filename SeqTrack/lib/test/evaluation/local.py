from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\got10k_lmdb'
    settings.got10k_path = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\lasot_extension_subset'
    settings.lasot_lmdb_path = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\lasot_lmdb'
    settings.lasot_path = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\lasot'
    settings.network_path = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\test/networks'    # Where tracking networks are stored.
    settings.nfs_path = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\nfs'
    settings.otb_path = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\OTB2015'
    settings.prj_dir = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack'
    settings.result_plot_path = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\test/result_plots'
    settings.results_path = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\test/tracking_results'    # Where to store tracking results
    settings.save_dir = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack'
    settings.segmentation_path = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\test/segmentation_results'
    settings.tc128_path = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\trackingnet'
    settings.uav_path = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\UAV123'
    settings.vot_path = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\VOT2019'
    settings.youtubevos_dir = ''

    return settings

