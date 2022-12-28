import sys
import torch
import facer

sys.path.append('..')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def face_parsing(img):
    image = facer.hwc2bchw(img).to(device=device)  # image: 1 x 3 x h x w

    face_detector = facer.face_detector('retinaface/mobilenet', device=device)
    with torch.inference_mode():
        faces = face_detector(image)

    # facer.show_bchw(facer.draw_bchw(image, faces))

    face_parser = facer.face_parser('farl/lapa/448', device=device)

    with torch.inference_mode():
        faces = face_parser(image, faces)

    seg_logits = faces['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w

    print(seg_probs.shape)
    # facer.show_bhw(x.argmax(dim=1).float()/seg_logits.size(1)*255)
    skin_nose = seg_probs[:, 1, :, :] + seg_probs[:, 6, :, :]
    return skin_nose.float() / seg_logits.size(1) * 255
    # facer.show_bhw(skin_nose.float()/seg_logits.size(1)*255)
    # facer.show_bchw(facer.draw_bchw(image, faces))


if __name__ == '__main__':
    img = facer.read_hwc('data/sideface.jpg')
    face_parsing(img)
