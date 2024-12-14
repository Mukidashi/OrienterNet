import argparse
import os, sys
import numpy as np 
import cv2 
import yaml 
import h5py

import matplotlib.pyplot as plt
import torch

from maploc.models.orienternet import OrienterNet
from maploc.utils.exif import EXIF
from maploc.demo import ImageCalibrator
from maploc.utils.geo import Projection, BoundaryBox
from maploc.osm.tiling import TileManager
from maploc.data.image import rectify_image, resize_image, pad_image
from maploc.models.voting import fuse_gps, argmax_xyr
from maploc.osm.viz import Colormap
from maploc.utils.viz_localization import likelihood_overlay, plot_dense_rotations
from maploc.utils.viz_2d import plot_images



def set_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, help="path to the model weights", default=None, required=True)
    parser.add_argument("--data_list",type=str, help="data list", default=None, required=True)
    parser.add_argument("--img_dir",type=str, help="path to the image directory", default=None, required=True)
    parser.add_argument('--out_dir',type=str,required=True)
    parser.add_argument('--calib_yaml',type=str, default=None)

    return parser


def read_data_list(dpath, img_dir):
    fp = open(dpath, "r")

    dataList = []

    fstr = next(fp,None)
    while(fstr):
        felems = fstr.strip().split()
        fstr = next(fp,None)

        ipath = os.path.join(img_dir,felems[0])
        lat = float(felems[1])
        lon = float(felems[2])

        dataList.append((ipath,lat,lon))

    return dataList


def yaml_construct_opencv_matrix(loader, node):
    mapping = loader.construct_mapping(node,deep=True)
    mat = np.array(mapping["data"])
    mat.resize(mapping["rows"], mapping["cols"])
    return mat


def read_calib_file(cpath): 

    yfp = open(cpath,"r")
    next(yfp,None)
    yaml.add_constructor(u'tag:yaml.org,2002:opencv-matrix', yaml_construct_opencv_matrix,yaml.SafeLoader)
    calib = yaml.safe_load(yfp)
    yfp.close()

    iwidth = calib["Camera.width"]
    iheight = calib["Camera.height"]
    fx = calib["Camera.fx"]
    fy = calib["Camera.fy"]
    cx = calib["Camera.cx"]
    cy = calib["Camera.cy"]

    Kmat = np.zeros((3,3))
    Kmat[0][0] = fx
    Kmat[1][1] = fy
    Kmat[0][2] = cx
    Kmat[1][2] = cy
    Kmat[2][2] = 1.0

    return Kmat, iwidth, iheight


def infer_data(model, image, camera, canvas, gravity,config):

    ## Prepare
    assert image.shape[:2][::-1] == tuple(camera.size.tolist())
    target_focal_length = config.data.resize_image / 2
    factor = target_focal_length / camera.f
    size = (camera.size * factor).round().int()

    image = torch.from_numpy(image).permute(2, 0, 1).float().div_(255)
    roll, pitch = gravity
    image, valid = rectify_image(image, camera.float(), roll=-roll, pitch=-pitch)

    image, _, camera, *maybe_valid = resize_image(image, size.tolist(), camera=camera, valid=valid)
    valid = None if valid is None else maybe_valid

    max_stride = max(model.image_encoder.layer_strides)
    size = (torch.ceil(size / max_stride) * max_stride).int()
    image, valid, camera = pad_image(image, size.tolist(), camera, crop_and_center=True)

    datas = {}
    datas["image"] = image
    datas["map"] = torch.from_numpy(canvas.raster).long()
    datas["camera"] = camera.float()
    datas["valid"] = valid
    datas_ = {k: v.to(device)[None] for k, v in datas.items()}


    ## Inference
    with torch.no_grad():
        pred = model(datas_)
    
    xy_gps = canvas.bbox.center
    uv_gps = torch.from_numpy(canvas.to_uv(xy_gps))

    lp_xyr = pred["log_probs"].squeeze(0)
    tile_size = canvas.bbox.size.min() / 2
    sigma = tile_size - 20
    lp_xyr = fuse_gps(lp_xyr, uv_gps.to(lp_xyr), 
                    config.model.pixel_per_meter, sigma=sigma) #Add GPS pos centered Prob
    # xyr = argmax_xyr(lp_xyr)

    prob = lp_xyr.exp().cpu()

    # print(prob.shape)

    return prob


def visualize_pose(img, prob, thresh=0.01, k=3, vec_size=15):
    t = torch.argmax(prob, -1)
    yaws = t.numpy() / prob.shape[-1] * 360
    prob = prob.max(-1).values / prob.max()
    mask = prob > thresh
    masked = prob.masked_fill(~mask, 0)
    max_ = torch.nn.functional.max_pool2d(
        masked.float()[None, None], k, stride=1, padding=k // 2
    )
    mask = (max_[0, 0] == masked.float()) & mask
    indices = np.where(mask.numpy() > 0)

    xy = indices[::-1]
    yaws = yaws[indices]
    for i in range(indices[0].shape[0]):
        theta = np.deg2rad(yaws[i])
        x = xy[0][i] + 0.5
        y = xy[1][i] + 0.5
        dx = vec_size * np.sin(theta)
        dy = -vec_size * np.cos(theta)
        img = cv2.arrowedLine(img, (int(x),int(y)), (int(x+dx),int(y+dy)), (0,0,0), 1, tipLength=0.3)
        img = cv2.circle(img, (int(x),int(y)), 2, (0,0,255), -1)
    
    return img


if __name__ == "__main__":

    parser = set_argparse()
    args = parser.parse_args()

    ROTATION_NUM = 256
    TILE_SIZE_METERS = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    #Set up model 
    ckpt = torch.load(args.weight,map_location=(lambda storage, loc: storage))
    config = ckpt["hyper_parameters"]
    config.model.update(num_rotations=ROTATION_NUM)
    config.model.image_encoder.backbone.pretrained = False

    model = OrienterNet(config.model).eval()
    
    state = {k[len("model.") :]: v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state, strict=True)
    model = model.to(device)

    calibrator = ImageCalibrator().to(device)

    Kmat, img_width, img_height = read_calib_file(args.calib_yaml)
    focal_length = (Kmat[0][0] + Kmat[1][1]) / 2.0


    os.makedirs(args.out_dir, exist_ok=True)

    ofp = open(os.path.join(args.out_dir,"out.list"),"w")

    #Proc 
    dataList = read_data_list(args.data_list,args.img_dir)
    for datas in dataList:
        ipath, lat, lon = datas

        print("###Processing: ", ipath)

        img = cv2.imread(ipath,cv2.IMREAD_COLOR)
        img = np.ascontiguousarray(img[:,:,::-1]) #BGR to RGB
        with open(ipath, "rb") as fid:
            exif = EXIF(fid, lambda: img.shape[:2])

        roll_pitch, camera = calibrator.run(img, focal_length, exif)
        print(" roll_pitch: ", roll_pitch)

        latlon = np.array([lat,lon])
        proj = Projection(*latlon)
        center = proj.project(latlon)
        bbox = BoundaryBox(center, center) + TILE_SIZE_METERS

        #OSM
        tiler = TileManager.from_bbox(proj, bbox + 10, config.data.pixel_per_meter)
        canvas = tiler.query(bbox)
        map_viz = Colormap.apply(canvas.raster)

        cent_lla = proj.unproject(canvas.bbox.center)
        map_size = canvas.bbox.size


        #Inference
        prob = infer_data(model, img, camera, canvas, roll_pitch, config)

        
        # Visualize for check  
        overlay = likelihood_overlay(prob.numpy().max(-1), map_viz.mean(-1,keepdims=True))*255
        overlay = visualize_pose(overlay, prob, vec_size=7.5*config.data.pixel_per_meter)

        bname, ext = os.path.splitext(os.path.basename(ipath))
        cpath = os.path.join(args.out_dir, bname + "_check.jpg")
        cv2.imwrite(cpath,overlay.astype(np.uint8))

        mpath = os.path.join(args.out_dir,bname+"_map.jpg")
        map_viz = map_viz[:,:,::-1]*255
        cv2.imwrite(mpath,map_viz.astype(np.uint8))
    
        fp = h5py.File(os.path.join(args.out_dir,bname+'.h5'), 'w')
        fp.create_dataset('prob', data=prob.numpy())

        ofp.write("%s %.08f %.08f %f %f %f\n" % (bname, cent_lla[0], cent_lla[1], map_size[0], map_size[1], config.data.pixel_per_meter))











