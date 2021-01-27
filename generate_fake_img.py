import cv2
import os
import numpy as np
import random


shape = ['circle', 'line', 'rectangle']


def draw_circle(img,color):
    center = ( random.randint(img.shape[1]//4,img.shape[1]//4*3),random.randint(img.shape[1]//4,img.shape[1]//4*3) )
    radius = random.randint(20,50)    
    cv2.circle(img, center, radius,color,-1)
    return img
    
def draw_line(img,color):
    piont1 = ( random.randint(0,img.shape[1]),random.randint(0,img.shape[0]) )
    w = random.randint(10,80)
    h = random.randint(10,80)
    piont2 = ( piont1[0]+w,piont1[1]+h ) 
    piont2 = ( random.randint(0,img.shape[1]),random.randint(0,img.shape[0]) )  
    draw_thickness = random.choice([3,4])   
    cv2.line(img, piont1, piont2, color,draw_thickness)
    return img
    
def draw_rectangle(img,color):
    piont1 = ( random.randint(0,img.shape[1]//2),random.randint(0,img.shape[0]//2) )
    w = random.randint(30,80)
    h = random.randint(30,80)
    piont2 = ( piont1[0]+w,piont1[1]+h )  
    cv2.rectangle(img, piont1, piont2, color,-1)
    return img

def DrawShape(img,shape,draw_color):
    if shape == 'circle' :
        img = draw_circle(img,draw_color)
    elif shape == 'line' :
        img = draw_line(img,draw_color)
    elif shape == 'rectangle' :
        img = draw_rectangle(img,draw_color)
    return img

def CreateDataSet(data_dir,image_size,split, number):
    for s in shape:
        if not os.path.exists(os.path.join(data_dir,split,s)) and number:
            os.makedirs(os.path.join(data_dir,split,s))
            
            for i in range(number):
                background_color = np.array([[[random.randint(0,255), random.randint(0,255),random.randint(0,255)]]])
                img = np.zeros([image_size,image_size,3]) + background_color
                draw_color = (random.randint(0,255), random.randint(0,255),random.randint(0,255))
                img = DrawShape(img,s,draw_color)
                name = s+'{:04d}.png'.format(i)
                cv2.imwrite(os.path.join(data_dir,split,s,name),img)

def generate_fake_image(data_dir, image_size, train_num = None,
                        valid_num = None, test_num = None):
    
    CreateDataSet(data_dir,image_size,'train',train_num)
    CreateDataSet(data_dir,image_size,'valid',valid_num)
    CreateDataSet(data_dir,image_size,'test',test_num)
    
        
if __name__ == '__main__':
    
    data_dir = r'./data'
    train_piece = 30
    valid_piece = 30
    test_piece = 30
    image_size = 224
    
    generate_fake_image(data_dir, image_size, train_piece, valid_piece, None)
        




