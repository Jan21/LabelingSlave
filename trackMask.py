from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
from pathlib import Path
from collections import namedtuple
import pickle

import cv2
import torch
import numpy as np
from glob import glob
import pickle

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)

import os
print(os.environ['DISPLAY'])


parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file', default='/home/jan/projects/episodic_unsupervised_learning/models/siammask_r50_l3/config.yaml')
parser.add_argument('--prelabel', type=bool, help='want to only prelabel init rects?', default=False)
parser.add_argument('--snapshot', type=str, help='model name', default='/home/jan/projects/episodic_unsupervised_learning/models/siammask_r50_l3/model.pth')
parser.add_argument('--video_name', type=str,
                    help='videos or image files', default='/home/jan/projects/episodic_unsupervised_learning/data/videos/Rose/VID_20191006_171135.mp4')
args = parser.parse_args()


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.pn*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def get_box_from_rect(img, rect_bbox):
    im_w = img.shape[1]
    im_h = img.shape[0]
    rect_w = rect_bbox[2]
    rect_h = rect_bbox[3]
    rect_x = max(0,rect_bbox[0])
    rect_y = max(0,rect_bbox[1])
    is_h_of_bbox_bigger = (rect_w < rect_h)
    # if the height of the rect is bigger then the width
    if is_h_of_bbox_bigger:
        # resize the width of the rect if there is a space
        if im_w >= rect_h:
            difference = rect_h - rect_w
            x_start = max(rect_x - difference/2, 0)
            x_end = min(x_start+rect_h,im_w)
            if x_end-x_start < rect_h:
               x_start = im_w - rect_h
            rect_x = x_start
            rect_w = rect_h
        # else make the box as big as width of the image
        else:
            rect_y = rect_y + (rect_h - rect_w)/2
            rect_h = im_w
            rect_x = 0
            rect_w = im_w
    # if the width is greater then height
    else:
        # resize the heigth of the rect if there is a space
        if im_h > rect_w:
            difference = rect_w - rect_h
            y_start = max(rect_y - difference/2, 0)
            y_end = min(y_start+rect_w,im_h)
            if y_end-y_start < rect_w:
               y_start = im_h - rect_w
            rect_y = y_start
            rect_h = rect_w        
        # else make the box as big as heigth of the image
        else:
            rect_x = rect_x + (rect_w - rect_h)/2
            rect_w = im_h
            rect_y = 0
            rect_h = im_h
    return [int(rect_x),int(rect_y),int(rect_w),int(rect_h)]
def crop_minAreaBox(img, bbox):
    bbox = get_box_from_rect(img,bbox)
    img_crop = img[ 
                       bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2],:]
    return img_crop

def crop_minAreaRect(img, polygon):

    # rotate img
    points = [[polygon[0],polygon[1]],[polygon[2],polygon[3]],[polygon[4],polygon[5]],[polygon[6],polygon[7]]]
    angle = np.rad2deg(np.arctan2(points[1][1] - points[0][1], points[1][0] - points[0][0]))
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))
    # rotate bounding box
    pts = cv2.transform(np.array([points]), M)
    pts[pts < 0] = 0
    # crop
    pts = pts[0]
    img_crop = img_rot[ 
                       pts[1][1]:pts[2][1],pts[0][0]:pts[1][0],:]
    return img_crop


def getTrackName():
    name = ""
    print('input track name, hit q to submit')
    while True:
        k = cv2.waitKey(0)
        if k == ord('q'):
            break                    
        name += chr(k)
        print(name)
    return name



def main():
    # load config
    prelabel = args.prelabel
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    # collect all vides
    videos_folder_str = '../data/videos/**/**.mp4'
    videos = glob(videos_folder_str)
    
    # iterate over all videos
    window_name = "tracker"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    collected_init_rects = []
    video_name = '/home/jan/projects/IRAFM/cvat_api/car_video' 
    path_to_video = Path(video_name)
    parent_folder = path_to_video.parent
    file_name = path_to_video.stem
    pickle_path = str(parent_folder/file_name)+".pkl"
    # load initial labels from pickled file or not
    do_load_from_pkl = False
    if os.path.isfile(pickle_path):
        with open(pickle_path,'rb') as f:
            init_data = pickle.load(f)
        do_load_from_pkl = True
    os.makedirs(parent_folder/file_name, exist_ok=True)
    first_frame = True
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    all_frames = []
    for frame in get_frames(video_name):
        all_frames.append({'frame':frame, 'annotations':{}})
    
    if os.path.isfile(f'{video_name}/annotations.pkl'):
        with open(f'{video_name}/annotations.pkl','rb') as f:
            frame_annotations = pickle.load(f)
        for i,annot in enumerate(frame_annotations):
            all_frames[i]['annotations'] = annot    

    def init_track(video_name, frame, annotations, tracker):
        try:
            init_rect = cv2.selectROI(video_name, frame, False, False)
            name = getTrackName()
            annotations[name] = init_rect 
        except:
            exit()
        tracker.init(frame, init_rect)
        return name
    
    def bakeOutFrame(frame, annotations):
        mask = np.zeros((frame.shape[0],frame.shape[1],3), np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for name, bbox in annotations.items():
            cv2.rectangle(mask, (bbox[0], bbox[1]),
                        (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                        (0, 255, 0), 3)
            
            cv2.putText(mask,name,(bbox[0], bbox[1]-20), font, .5,(255,255,255),2,cv2.LINE_AA)
            frame = cv2.add(frame, mask)
        return frame

    def delete_annotations(subarray,name_to_delete):
        for f in subarray:
            if name_to_delete in f['annotations']:
                del f['annotations'][name_to_delete]


    def on_trackbar(frame_ix):
        frame_data = all_frames[frame_ix]
        frame = bakeOutFrame(frame_data['frame'],frame_data['annotations'])
        cv2.imshow(video_name, frame)

    cv2.createTrackbar('trackbar',video_name,0,len(all_frames),on_trackbar)
    i = 0
    while i < len(all_frames):
        frame_data = all_frames[i]
        frame = frame_data['frame']
        annotations = frame_data['annotations']
        if first_frame:
            key = cv2.waitKey(0)
            i = cv2.getTrackbarPos('trackbar',video_name)
            frame_data = all_frames[i]
            name = init_track(video_name, frame_data['frame'], \
                frame_data['annotations'],tracker)                
            first_frame = False
        else:
            outputs = tracker.track(frame)
            if 'bbox' in outputs:
                bbox = np.array(outputs['bbox']).astype(np.int32)
                annotations[name] = bbox
                frame = bakeOutFrame(frame,annotations)
            cv2.imshow(video_name, frame)
            key = cv2.waitKey(40)
            if key == 32:
                while True:
                    print('hit c to create new track or d to delete track')
                    key2 = cv2.waitKey(0)
                    i = cv2.getTrackbarPos('trackbar',video_name)
                    if key2 == 99: # 'c' was pressed
                        frame_data = all_frames[i]
                        name = init_track(video_name, frame_data['frame'], \
                            frame_data['annotations'],tracker)
                        break
                    if key2 == 100: # 'd' was pressed
                        name_to_delete = getTrackName()
                        print('hit d to mark end frame for deletion')
                        key3 = cv2.waitKey(0)
                        j = cv2.getTrackbarPos('trackbar',video_name)
                        if key2 == 100: # 'd' was pressed
                            delete_annotations(all_frames[i:j+1],name_to_delete)
        i += 1
        if i > len(all_frames):
            break
        cv2.setTrackbarPos('trackbar',video_name,i)
    
    frame_annotations = []
    for i,frame_data in enumerate(all_frames):
        frame_annotations.append(frame_data["annotations"])
    with open(f'{video_name}/annotations.pkl','wb') as f:
        pickle.dump(frame_annotations,f)
    


if __name__ == '__main__':
    main()