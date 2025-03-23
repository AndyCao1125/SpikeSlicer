from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_path = '/home/dataset/GOT10K'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = ''
    settings.lasot_path = ''
    settings.network_path = ''
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.oxuva_path = ''
    settings.result_plot_path = ''
    settings.results_path = ''    # Where to store tracking results
    settings.segmentation_path = ''
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''
    settings.fe108_dir = '/home/dataset/FE108'

    return settings

