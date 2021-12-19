import matplotlib.pyplot as plt
import cv2

def scale_to_coords(x1,y1,x2,y2,orig_shape):
    x1 = x1 * (orig_shape[1]/410.0)
    x2 = x2 * (orig_shape[1]/410.0)
    
    y1 = y1 * (orig_shape[0]/410.0)
    y2 = y2 * (orig_shape[0]/410.0)

    return x1,y1,x2,y2

def plot_img_with_boxes(image,bboxes,thickness = 2, color = (255,0,0)):
    plt.figure(figsize = (15,13))
    for i in range(bboxes.shape[0]):
        start_pt = (int(bboxes[i][0]),int(bboxes[i][1]))
        end_pt = (int(bboxes[i][2]),int(bboxes[i][3]))
        image = cv2.rectangle(image,start_pt,end_pt,color,thickness)
    plt.imshow(image)
    plt.show()

def plot_pred_gt_side_by_side(eval_dataset,
                              predictions,
                              color1 = (255,0,0),
                              color2 = (0,0,255),
                              color3 = (0,255,0),
                              color4 = (247, 243, 15),
                              thickness = 2,
                              judgements = None):
    
    for idx in range(len(eval_dataset)):
        image, target = eval_dataset[idx]
        gtboxes = target["boxes"]
        bboxes = predictions[idx]
        image = image.permute(1,2,0).numpy()
        image1 = image.copy()
        image2 = image.copy()
        
        fig, axs = plt.subplots(1,2,figsize = (20,30),sharex = True)
        for i in range(bboxes.shape[0]):
            start_pt = (int(bboxes[i][0]),int(bboxes[i][1]))
            end_pt = (int(bboxes[i][2]),int(bboxes[i][3]))
            if judgements:
                if judgements[idx][i] == 0:
                    image1 = cv2.rectangle(image1,start_pt,end_pt,color3,thickness)
                elif judgements[idx][i] == -1:
                    image1 = cv2.rectangle(image1,start_pt,end_pt,color4,thickness)
                else:
                    image1 = cv2.rectangle(image1,start_pt,end_pt,color1,thickness)
            else:
                image1 = cv2.rectangle(image1,start_pt,end_pt,color1,thickness)

        for i in range(gtboxes.shape[0]):
            start_pt = (int(gtboxes[i][0]),int(gtboxes[i][1]))
            end_pt = (int(gtboxes[i][2]),int(gtboxes[i][3]))
            image2 = cv2.rectangle(image2,start_pt,end_pt,color2,thickness)

        axs[0].imshow(image1)
        axs[1].imshow(image2)
    
def display(image,fig_size = (10,8)):
    plt.figure(figsize = fig_size)
    plt.imshow(image)
    plt.show()