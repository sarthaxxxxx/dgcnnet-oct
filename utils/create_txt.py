import os
import csv

def __filegen__(root, mode, save):
	r"""Create .txt files storing paths of data. To be used by the dataloader.
	Splits : train, val, and test.
	train :  2975
	val : 500
	test : 1525 
	"""
	images_path = os.path.join(root, 'leftImg8bit')
	gt_path = os.path.join(root, 'gtFine')

	img_paths, gt_paths = [], [] 

	for idx in sorted(os.listdir(os.path.join(images_path, mode))):
		for jdx in sorted(os.listdir(os.path.join(images_path, mode + '/' + idx))):
			img_paths.append(os.path.join(images_path, mode + '/' + idx + '/' + jdx))
			gt_paths.append(os.path.join(gt_path, mode + '/' + idx + '/' + jdx.replace(jdx.split('_')[3], 'gtFine_labelIds.png')))

	try:
		with open(os.path.join(save, mode + '.txt'), 'w') as f:
			[f.write(str(img_paths[index]) + '\t' + str(gt_paths[index]) + '\n') 
			for index in range(len(img_paths))]
			f.close()
	except ValueError:
		print(f'{mode} file missing!')	
	'''
	try:
		with open(os.path.join(save, mode + '.csv'), 'w') as f:
			writer = csv.writer(f, delimiter = '\t')
			writer.writerows(zip(img_paths, gt_paths))
	except ValueError:
		print(f'{mode} file missing!')	
	'''

if __name__=='__main__':
	for mode in ['train', 'val', 'test']:
		__filegen__('./dgcn/city_dataset/', mode, './dgcn/data/')
