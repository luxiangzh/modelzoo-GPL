from pycocotools.coco import COCO
import shutil
import os

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def get_classes_and_index(path):
    D = {}
    f = open(path)
    for line in f:
        temp = line.rstrip().split(',', 2)
        print("temp[0]:" + temp[0] + "\n")
        print("temp[1]:" + temp[1] + "\n")
        D[temp[1]] = temp[0]
    return D

def coco2yolo(dataType):
    annFile = './annotations/instances_%s.json' % dataType
    classes = get_classes_and_index('./coco_class.txt')

    if not os.path.exists('./images'):
        os.makedirs('./images')

    os.symlink(os.path.abspath(dataType), './images/%s' % dataType)

    if not os.path.exists('./labels/%s' % dataType):
        os.makedirs('./labels/%s' % dataType)
    else:
        shutil.rmtree('./labels/%s' % dataType)
        os.makedirs('./labels/%s' % dataType)

    coco = COCO(annFile)
    list_file = open('%s.txt' % dataType, 'w')

    imgIds = coco.getImgIds()
    catIds = coco.getCatIds()

    for imgId in imgIds:
        objCount = 0
        Img = coco.loadImgs(imgId)[0]
        filename = Img['file_name']
        width = Img['width']
        height = Img['height']
        annIds = coco.getAnnIds(imgIds=imgId, catIds=catIds, iscrowd=None)
        for annId in annIds:
            anns = coco.loadAnns(annId)[0]
            catId = anns['category_id']
            cat = coco.loadCats(catId)[0]['name']

            if cat in classes:
                objCount = objCount + 1
                out_file = open('labels/%s/%s.txt' % (dataType, filename[:-4]), 'a')
                cls_id = classes[cat]
                box = anns['bbox']
                size = [width, height]
                bb = convert(size, box)
                out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
                out_file.close()

        list_file.write('./images/%s/%s\n' % (dataType, filename))

    list_file.close()

if __name__ == '__main__':
    coco2yolo('train2017')
    coco2yolo('val2017')