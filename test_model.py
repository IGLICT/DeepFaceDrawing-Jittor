import time
from CombineModel_jt import CombineModel
import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
import jittor as jt

jt.flags.use_cuda = 1

#models for face/eye1/eye2/nose/mouth
combine_model = CombineModel()

print('start')
fileRoot = './test'
images_path = sorted(glob.glob(fileRoot+r"/*"))

params = [
    [0.80, 0.63, 1.0, 0.88, 0.93, 1],
    [1.0, 1.0, 1.0, 1.0, 0.84, 0],
    [0.1, 0.39, 0.58, 0.63, 0.49, 1],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1],
    [0.78, 1.0, 1.0, 1.0, 0.79, 1]
]

i = 0

for x,fileName in enumerate(images_path):
    #fileName = fileRoot + str(x) + "_out_Similarity.jpg"
    print(fileName)
    mat_img = cv2.imread(fileName)
    mat_img = cv2.resize(mat_img, (512, 512), interpolation=cv2.INTER_CUBIC)
    mat_img = cv2.cvtColor(mat_img, cv2.COLOR_RGB2BGR)
    sketch = (mat_img).astype(np.uint8)
    combine_model.sex = params[i][5]
    #666
    combine_model.part_weight['eye1'] = params[i][0]
    combine_model.part_weight['eye2'] = params[i][1]
    combine_model.part_weight['nose'] = params[i][2]
    combine_model.part_weight['mouth'] = params[i][3]
    combine_model.part_weight[''] = params[i][4]
    
    print(mat_img.shape)
    combine_model.predict_shadow(mat_img)
    
    print(combine_model.generated)
    cv2.imwrite('ori'+ str(x) +'.jpg',cv2.cvtColor(combine_model.generated, cv2.COLOR_BGR2RGB))
    i = i + 1
    jt.gc()
