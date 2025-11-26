from utils.mnist import extract_training, display_image


res = extract_training()
images, labels, mndata = res
display_image(images, labels, mndata)