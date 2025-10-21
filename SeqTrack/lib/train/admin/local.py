# class EnvironmentSettings:
#     def __init__(self):
#         self.workspace_dir = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack'    # Base directory for saving network checkpoints.
#         self.tensorboard_dir = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\tensorboard'    # Directory for tensorboard files.
#         self.pretrained_networks = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\pretrained_networks'
#         self.lasot_dir = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\lasot'
#         self.got10k_dir = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\got10k'
#         self.lasot_lmdb_dir = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\lasot_lmdb'
#         self.got10k_lmdb_dir = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\got10k_lmdb'
#         self.trackingnet_dir = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\trackingnet'
#         self.trackingnet_lmdb_dir = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\trackingnet_lmdb'
#         self.coco_dir = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\coco'
#         self.coco_lmdb_dir = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\coco_lmdb'
#         self.imagenet1k_dir = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\imagenet1k'
#         self.imagenet22k_dir = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\imagenet22k'
#         self.lvis_dir = ''
#         self.sbd_dir = ''
#         self.imagenet_dir = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\vid'
#         self.imagenet_lmdb_dir = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\vid_lmdb'
#         self.imagenetdet_dir = ''
#         self.ecssd_dir = ''
#         self.hkuis_dir = ''
#         self.msra10k_dir = ''
#         self.davis_dir = ''
#         self.youtubevos_dir = ''

class EnvironmentSettings:
    def __init__(self):
        base = r"D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack"

        self.workspace_dir = base  # Base directory for saving network checkpoints.
        self.tensorboard_dir = rf"{base}\tensorboard"  # Directory for tensorboard files.
        self.pretrained_networks = rf"{base}\pretrained_networks"

        data = rf"{base}\data"
        self.lasot_dir = rf"{data}\lasot"
        self.got10k_dir = rf"{data}\got10k"
        self.lasot_lmdb_dir = rf"{data}\lasot_lmdb"
        self.got10k_lmdb_dir = rf"{data}\got10k_lmdb"
        self.trackingnet_dir = rf"{data}\trackingnet"
        self.trackingnet_lmdb_dir = rf"{data}\trackingnet_lmdb"
        self.coco_dir = rf"{data}\coco"
        self.coco_lmdb_dir = rf"{data}\coco_lmdb"
        self.imagenet1k_dir = rf"{data}\imagenet1k"
        self.imagenet22k_dir = rf"{data}\imagenet22k"
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = rf"{data}\vid"
        self.imagenet_lmdb_dir = rf"{data}\vid_lmdb"
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
