import torch
import module1 as m1
import module2 as m2
import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import pickle
import time

cap = cv.VideoCapture(1)
cap.set(cv.CAP_PROP_EXPOSURE, -6)

inp=torch.tensor([0.4904496967792511, 0.7314993143081665, 3.4411408478263183e-07, 0.5491338968276978, 0.7282752394676208, -0.02637900784611702, 0.6041166186332703, 0.6826633214950562, -0.03456651419401169, 0.6429800391197205, 0.6388425827026367, -0.039022885262966156, 0.6668127775192261, 0.6022310256958008, -0.04518488422036171, 0.5642224550247192, 0.5887894034385681, -0.0282245222479105, 0.5833373069763184, 0.5156227350234985, -0.03933302313089371, 0.5925359725952148, 0.4662647247314453, -0.049954477697610855, 0.5981656908988953, 0.42438119649887085, -0.05944734066724777, 0.5265827178955078, 0.5760579109191895, -0.026261145249009132, 0.5334091782569885, 0.488240122795105, -0.03731604665517807, 0.5376883745193481, 0.43612533807754517, -0.05279893800616264, 0.5381519794464111, 0.39590704441070557, -0.06573648750782013, 0.48933470249176025, 0.5820584893226624, -0.025778217241168022, 0.4880930483341217, 0.5038098096847534, -0.04029923304915428, 0.4889899492263794, 0.454898864030838, -0.05561922863125801, 0.48949652910232544, 0.41530072689056396, -0.06679302453994751, 0.4560539126396179, 0.6009379625320435, -0.02685721032321453, 0.4473109841346741, 0.5456418991088867, -0.04205792397260666, 0.44264543056488037, 0.5087171792984009, -0.050622910261154175, 0.4397544860839844, 0.4769922196865082, -0.056364864110946655])
model = torch.load('./mdl/best_model.ckp')
model.eval()
output=model(inp)
print(output)



while True:
    
    
    gesture='openplm_test'
    
    while True:
        try:
            normal=[]
            ret, framen = cap.read()
            i=m2.handlandmarker(framen)
            
            for a in i.hand_landmarks[0]:     
                normal.append(a.x)
                normal.append(a.y)
                normal.append(a.z)

            res=model(torch.tensor(normal))
            print(res)
            # STEP 5: Process the classification result. In this case, visualize it.
            annotated_image = m2.draw_landmarks_on_image(framen, i,res,['ok','openplm','thumup'])
            
            cv.imshow("tracked",annotated_image)

            break
        except Exception as e:
        
            print(f"An error occurred: {e}")

        
    c = cv.waitKey(1)
    if c == 27:
        break


cap.release()
cv.destroyAllWindows()