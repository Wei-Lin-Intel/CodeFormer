import os
import cv2
import time
import argparse
import glob
import numpy as np
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
#from basicsr.utils.misc import gpu_is_available, get_device
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray

from basicsr.utils.registry import ARCH_REGISTRY
import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu.graphs as htgraphs
from habana_frameworks.torch.utils.experimental import detect_recompilation_auto_model

pretrain_model_url = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
}

def set_realesrgan():
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.realesrgan_utils import RealESRGANer

    use_half = False
    #if torch.cuda.is_available(): # set False in CPU/MPS mode
    #    no_half_gpu_list = ['1650', '1660'] # set False for GPUs that don't support f16
    #    if not True in [gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list]:
    #        use_half = True

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
    upsampler = RealESRGANer(
        scale=2,
        model_path="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
        model=model,
        tile=args.bg_tile,
        tile_pad=40,
        pre_pad=0,
        half=use_half
    )

    #if not gpu_is_available():  # CPU
    #    import warnings
    #    warnings.warn('Running on CPU now! Make sure your PyTorch version matches your CUDA.'
    #                    'The unoptimized RealESRGAN is slow on CPU. '
    #                    'If you want to disable it, please remove `--bg_upsampler` and `--face_upsample` in command.',
    #                    category=RuntimeWarning)
    return upsampler

if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('hpu')
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', type=str, default='./512k_images',
            help='Input image, video or folder. Default: inputs/whole_imgs')
    parser.add_argument('-o', '--output_path', type=str, default='./output_images',
            help='Output folder. Default: results/<input_name>_<w>')
    parser.add_argument('-w', '--fidelity_weight', type=float, default=0.5,
            help='Balance the quality and fidelity. Default: 0.5')
    parser.add_argument('-s', '--upscale', type=int, default=2,
            help='The final upsampling scale of the image. Default: 2')
    parser.add_argument('--has_aligned', action='store_true', help='Input are cropped and aligned faces. Default: False')
    parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face. Default: False')
    parser.add_argument('--draw_box', action='store_true', help='Draw the bounding box for the detected faces. Default: False')
    # large det_model: 'YOLOv5l', 'retinaface_resnet50'
    # small det_model: 'YOLOv5n', 'retinaface_mobile0.25'
    parser.add_argument('--detection_model', type=str, default='retinaface_resnet50',
            help='Face detector. Optional: retinaface_resnet50, retinaface_mobile0.25, YOLOv5l, YOLOv5n, dlib. \
                Default: retinaface_resnet50')
    parser.add_argument('--bg_upsampler', type=str, default='None', help='Background upsampler. Optional: realesrgan')
    parser.add_argument('--face_upsample', action='store_true', help='Face upsampler after enhancement. Default: False')
    parser.add_argument('--bg_tile', type=int, default=400, help='Tile size for background sampler. Default: 400')
    parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored faces. Default: None')
    parser.add_argument('--save_video_fps', type=float, default=None, help='Frame rate for saving video. Default: None')
    parser.add_argument('--bf16', action='store_true', help='Mixed Precision for CodeFormer. Default: False')
    parser.add_argument('--graph', action='store_true', help='Graph Mode for CodeFormer. Default: False')

    args = parser.parse_args()

    # ------------------------ input & output ------------------------
    w = args.fidelity_weight
    input_video = False
    if args.input_path.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')): # input single img path
        input_img_list = [args.input_path]
        result_root = f'results/test_img_{w}'
    elif args.input_path.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')): # input video path
        from basicsr.utils.video_util import VideoReader, VideoWriter
        input_img_list = []
        vidreader = VideoReader(args.input_path)
        image = vidreader.get_frame()
        while image is not None:
            input_img_list.append(image)
            image = vidreader.get_frame()
        audio = vidreader.get_audio()
        fps = vidreader.get_fps() if args.save_video_fps is None else args.save_video_fps
        video_name = os.path.basename(args.input_path)[:-4]
        result_root = f'results/{video_name}_{w}'
        input_video = True
        vidreader.close()
    else: # input img folder
        if args.input_path.endswith('/'):  # solve when path ends with /
            args.input_path = args.input_path[:-1]
        # scan all the jpg and png images
        input_img_list = sorted(glob.glob(os.path.join(args.input_path, '*.[jpJP][pnPN]*[gG]')))
        result_root = f'results/{os.path.basename(args.input_path)}_{w}'

    if not args.output_path is None: # set output path
        result_root = args.output_path

    test_img_num = len(input_img_list)
    if test_img_num == 0:
        raise FileNotFoundError('No input image/video is found...\n'
            '\tNote that --input_path for video should end with .mp4|.mov|.avi')

    # ------------------ set up background upsampler ------------------
    if args.bg_upsampler == 'realesrgan':
        bg_upsampler = set_realesrgan()
    else:
        bg_upsampler = None

    # ------------------ set up face upsampler ------------------
    if args.face_upsample:
        if bg_upsampler is not None:
            face_upsampler = bg_upsampler
        else:
            face_upsampler = set_realesrgan()
    else:
        face_upsampler = None

    # ------------------ set up CodeFormer restorer -------------------
    net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                                            connect_list=['32', '64', '128', '256'])#.to(device)

    # ckpt_path = 'weights/CodeFormer/codeformer.pth'
    ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'],
                                    model_dir='weights/CodeFormer', progress=True, file_name=None)
    checkpoint = torch.load(ckpt_path)['params_ema']
    net.load_state_dict(checkpoint)
    if args.bf16:
        net = net.bfloat16()
    net.eval()
    if args.graph:
        net = htgraphs.wrap_in_hpu_graph(net)
    #net = detect_recompilation_auto_model(net)
    net = net.to(device)
    startEv =ht.hpu.Event(enable_timing=True)
    endEv = ht.hpu.Event(enable_timing=True)

    # ------------------ set up FaceRestoreHelper -------------------
    # large det_model: 'YOLOv5l', 'retinaface_resnet50'
    # small det_model: 'YOLOv5n', 'retinaface_mobile0.25'
    if not args.has_aligned:
        print(f'Face detection model: {args.detection_model}')
    if bg_upsampler is not None:
        print(f'Background upsampling: True, Face upsampling: {args.face_upsample}')
    else:
        print(f'Background upsampling: False, Face upsampling: {args.face_upsample}')

    face_helper = FaceRestoreHelper(
        args.upscale,
        face_size=512,
        crop_ratio=(1, 1),
        det_model = args.detection_model,
        save_ext='png',
        use_parse=True,
        device=device)

    face_parser_time = []
    code_former_time = []
    face_detector_time = []
    e2e_time = []

    # -------------------- start to processing ---------------------
    for i, img_path in enumerate(input_img_list):

        start = time.time()
        # clean all the intermediate results to process the next image
        face_helper.clean_all()

        if isinstance(img_path, str):
            img_name = os.path.basename(img_path)
            basename, ext = os.path.splitext(img_name)
            print(f'[{i+1}/{test_img_num}] Processing: {img_name}')
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        else: # for video processing
            basename = str(i).zfill(6)
            img_name = f'{video_name}_{basename}' if input_video else basename
            print(f'[{i+1}/{test_img_num}] Processing: {img_name}')
            img = img_path

        if args.has_aligned:
            # the input faces are already cropped and aligned
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            face_helper.is_gray = is_gray(img, threshold=10)
            if face_helper.is_gray:
                print('Grayscale input: True')
            face_helper.cropped_faces = [img]
        else:
            face_helper.read_image(img)
            # get face landmarks for each face
            t1 = time.time()
            num_det_faces = face_helper.get_face_landmarks_5(
                only_center_face=args.only_center_face, resize=640, eye_dist_threshold=5)
            t2 = time.time()
            print("Face Parser Infer Latency: {:.2f} ms".format((t2 - t1) * 1000))
            face_parser_time.append((t2 - t1) * 1000)
            print(f'\tdetect {num_det_faces} faces')
            # align and warp each face
            face_helper.align_warp_face()

        activities = [torch.profiler.ProfilerActivity.CPU]
        activities.append(torch.profiler.ProfilerActivity.HPU)

        # face restoration for each cropped face
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            # prepare data
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).contiguous(memory_format=torch.channels_last)

            try:
                with torch.no_grad(), torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=args.bf16):

                    if i > 4:
                        with torch.profiler.profile(
                            schedule=torch.profiler.schedule(wait=0, warmup=20, active=5, repeat=10),
                            activities=activities,
                            on_trace_ready=torch.profiler.tensorboard_trace_handler('logs')) as profiler:
                            for i in range(100):
                                output = net(cropped_face_t.to(device, non_blocking = True), w=w, adain=True)[0]
                                output.to('cpu')
                                htcore.mark_step()
                                profiler.step()
                        exit()
                    else:
                        output = net(cropped_face_t.to(device, non_blocking = True), w=w, adain=True)[0]
            except Exception as error:
                print(f'\tFailed inference for CodeFormer: {error}')

