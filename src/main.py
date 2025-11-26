from utils.mnist import *

res = extract_training()
images, labels, mndata = res
display_image(images, labels, mndata)