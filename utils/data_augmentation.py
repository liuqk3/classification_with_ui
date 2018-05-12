from skimage import  io,exposure, img_as_float
import matplotlib.pyplot as plt
import os
root_dir='/home/yangwf/code/AttributePrediction/data/train_valid/'
category_dir=['skirt_length_labels','coat_length_labels','collar_design_labels','lapel_design_labels','neck_design_labels','neckline_design_labels','pant_length_labels','sleeve_length_labels']
m=0
for category in category_dir:
	modes=os.listdir(os.path.join(root_dir,category))
	for mode in modes:
		labels_list=os.listdir(os.path.join(root_dir,category,mode))
		for label in labels_list:
			images_names=os.listdir(os.path.join(root_dir,category,mode,label))
			for image in images_names:
				image_full_path=os.path.join(root_dir,category,mode,label,image)
				image=io.imread(image_full_path)
				image = img_as_float(image)
				gam1= exposure.adjust_gamma(image, 2)   
				gam2= exposure.adjust_gamma(image, 0.5)  
				io.imsave(image_full_path[:-4]+'_1.jpg',gam1)
				io.imsave(image_full_path[:-4]+'_2.jpg',gam2)
				m=m+2
				if m%200==0:
					print 'processed {} images'.format(m)