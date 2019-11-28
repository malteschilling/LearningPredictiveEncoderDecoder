import numpy as np
import cv2

height, width = 32, 32
sequ_length = 5
number_sequences = 8000
number_test = number_sequences/4
block_half_dim = np.array([4, 4])
max_velocity = 8
const_acc = 2 

class FallingObject(object):
    """
    Simple Dynamic for a falling update (position, velocity, acceleration).
    
    Provides an update step - new position and velocity.
    """
    def __init__(self, x_pos = 32, y_pos = 8, vel = 0):
        # Attributes of the falling block: 
        #   pos_x = constant, set randomly
        #   pos_y = falling
        #   vel = increasing, randomly initialised
        #   acc = const set to 2 (down is > 0 in openCV)
        self.dynamic_attributes = np.array([x_pos, y_pos, vel, const_acc])
        self.vel_upd = 8
        self.time = 0
        #print("Init: ", x_pos, y_pos, vel)
        
    def update_object(self):
        # We just assume acceleration over time,
        # for movement distance: for next timestep current velocity is assumed
        self.dynamic_attributes[1] += self.dynamic_attributes[2]
        #self.dynamic_attributes[1] += 0.5 * self.dynamic_attributes[3] * self.time**2
        self.dynamic_attributes[2] += self.dynamic_attributes[3]
        self.time+=1
        return self.dynamic_attributes
        #print(self.vel_upd, self.dynamic_attributes[1])

# Create DataSet
dataset_imgs = np.zeros((number_sequences, sequ_length, height, width), np.uint8)
dataset_gt = np.zeros((number_sequences, sequ_length, 4), np.uint8)
for sequ_it in range(0, number_sequences ):
    block = FallingObject( np.random.randint( block_half_dim[0], (width - block_half_dim[0]) ), 
#        np.random.randint( block_half_dim[1], (height/2 - block_half_dim[1]) ),
        np.random.randint( block_half_dim[1], (height/2 - block_half_dim[1]) ),
#        np.random.randint( -const_acc, (max_velocity -const_acc + 1) ) )
        np.random.randint( 0, (max_velocity + 1) ) )
    for i in range(0,sequ_length):
        #cv2.circle(img,(10, 19), 5, (255), -1)
        #img = np.zeros((height, width), np.uint8)
        dataset_gt[sequ_it][i] = block.update_object()
        # Use rectangle shape
        #cv2.rectangle( dataset_imgs[sequ_it][i], (block.dynamic_attributes[0]-5, block.dynamic_attributes[1]-5), 
         #   (block.dynamic_attributes[0]+5, block.dynamic_attributes[1]+5), (255), -1)
        # Use circle shape
        cv2.circle( dataset_imgs[sequ_it][i], (block.dynamic_attributes[0], block.dynamic_attributes[1]), 5, (255), -1)
        #cv2.imshow("Show falling Sprite", img)
        #k = cv2.waitKey(0)
        cv2.imwrite("sprites/square_circle_opencv_" + str(sequ_it).zfill(4) + "_" + str(i) + ".png", dataset_imgs[sequ_it][i])

# For sequence length 1       
#x_train = dataset_imgs[:-number_test,0,:,:].reshape(dataset_imgs.shape[0]-number_test,1,dataset_imgs.shape[2],dataset_imgs.shape[3])
#x_test = dataset_imgs[-number_test:,0,:,:].reshape(number_test,1,dataset_imgs.shape[2],dataset_imgs.shape[3])
x_train = dataset_imgs[:-number_test,:-1,:,:].reshape(dataset_imgs.shape[0]-number_test,dataset_imgs.shape[1]-1,dataset_imgs.shape[2],dataset_imgs.shape[3])
x_test = dataset_imgs[-number_test:,:-1,:,:].reshape(number_test,dataset_imgs.shape[1]-1,dataset_imgs.shape[2],dataset_imgs.shape[3])

y_train = dataset_imgs[:-number_test,1:,:,:].reshape(dataset_imgs.shape[0]-number_test,dataset_imgs.shape[1]-1,dataset_imgs.shape[2],dataset_imgs.shape[3])
y_test = dataset_imgs[-number_test:,1:,:,:].reshape(number_test,dataset_imgs.shape[1]-1,dataset_imgs.shape[2],dataset_imgs.shape[3])
print(x_train.shape, y_test.shape)

import pickle
with open('dataset_falling_circles_seq', 'wb') as file_pi:
    pickle.dump( [x_train, x_test, y_train, y_test, dataset_gt], file_pi) 