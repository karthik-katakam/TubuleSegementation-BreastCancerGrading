from utils import *
import json, time
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"]=""
model = r2_unet(256, 256, 1)
model.compile(tf.keras.optimizers.Adam(learning_rate=2e-4), loss="binary_crossentropy", metrics=[tf.keras.metrics.MeanIoU(num_classes=2), f1_metric])
model.load_weights('./model.h5')
#T_95_images = ['10_45.jpg', '11_29.jpg', '11_37.jpg', '12_19.jpg', '9_20.jpg']
T_95_images = sorted(os.listdir('/home/kmit/kd_tf_testing/Tubule/ViewerTrails/sampleData/fL8Tiles_992/filtered/'))
print("Done")

json_ = {'description': "",  
         'name': "NewAlgoV2",
         'elements': []}



### np.std() > 12 - 15 
image_size_ = 992
ckt = 0
lst = []
start = time.time()
for image in T_95_images:
    lst.append(image)
    img = cv2.imread('/home/kmit/kd_tf_testing/Tubule/ViewerTrails/sampleData/fL8Tiles_992/filtered/'+image)
    if(np.std(img) > 12):
        #if(ckt > 5): break
        pred_mask = model.predict(img.reshape(1,256,256,3)).reshape(256,256)
        _,pred_mask = cv2.threshold(pred_mask,0.4,1,cv2.THRESH_BINARY)
        pred_mask *= 255
        #pred_mask = pred_mask.astype(np.uint8)
        pred_mask = Image.fromarray(np.uint8(pred_mask))
        pred_mask = pred_mask.resize((image_size_, image_size_))
        pred_mask = np.array(pred_mask)
        pred_mask = pred_mask.reshape((image_size_, image_size_, 1))
        #cv2.imwrite("pred_" + image, pred_mask)
        contours, _ = cv2.findContours(image=pred_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        hulls = [cv2.convexHull(contours[i], False) for i in range(len(contours))]
        p = image.split("_")
        p[1] = p[1].split(".")[0]
        position=[int(p[0]), int(p[1])]
        if len(hulls) != 0:
            for hull in hulls:
                if(hull.shape[0] > 10):
                    ckt += 1
                    hull.resize((hull.shape[0], 2))
                    hull[:, 0] += (image_size_)*(position[0])
                    hull[:, 1] += (image_size_)*(position[1])
                    hull = np.pad(hull, (0, 1), mode = 'constant')[:-1]
                    json_['elements'].append({"fillColor": "rgba(0,0,0,0)",
                            "lineColor": "rgb(0,0,0)",
                            "group": "TubuleSegmentation",
                            "lineWidth": 2,
                            "type": "polyline",
                            "closed": True, 
                            "points": hull.tolist()})
                    
            
with open("T95_996_resized_praneeth_5_test_newAlgo.json", "w") as outfile:  json.dump(json_, outfile)
print('Tubules: ', ckt)
print(time.time() - start)
