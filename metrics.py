import torch

@torch.no_grad()
def validate_voc_format(model,eval_dataset,top_k = 1,plot = False,apply_brisk = False):
    print("Starting evaluation .....\n")
    model.eval()
    stats = []
    predictions = []
    ignoreable = []
    wrong_recogs = []
    for idx in tqdm(range(len(eval_dataset))):
        img_ts,target_ts = eval_dataset[idx]
        
        img_path = os.path.join(eval_dataset.root, "images", eval_dataset.imgs[idx])
        img_actual = cv2.imread(img_path)
        img_actual = cv2.cvtColor(img_actual,cv2.COLOR_BGR2RGB)
        img_actual = img_actual / 255.0
        img_actual = torch.from_numpy(img_actual).float()
        img_actual = img_actual.permute(2,0,1)

        img_actual = img_actual.to(GPU)
        img_ts = img_ts.to(GPU)

        labels = target_ts["annots"]
        #labels = target_ts["labels"]
        
        uniq_labels = set(labels)
        pred_uniq_labels = set([])

        prediction_boxes = model([img_ts])[0]["boxes"].to(CPU)
        n_boxes_pred = prediction_boxes.shape[0]
        
        iou_matrix = ops.box_iou(prediction_boxes,target_ts["boxes"])
        best_ious, best_iou_indices = iou_matrix.max(1)

        correct = torch.zeros(n_boxes_pred)
        marker = torch.zeros(n_boxes_pred)
        
        ignore = 0
        
        for i in range(len(best_ious)):     # processing one image at a time
            if best_ious[i] <= 0.08:
                ignore +=1

            if best_ious[i] >= 0.50:
                
                x1,y1,x2,y2 = prediction_boxes[i]
                x1,y1,x2,y2 = scale_to_coords(x1,y1,x2,y2,img_actual.shape[1:])

                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                
                cropped = img_actual[:,y1:y2,x1:x2]
                #cropped = img_ts[:,y1:y2,x1:x2]
                pcls = yield_top_k_matches(cropped,k = top_k,apply_BRISK = apply_brisk)
                
                if labels[best_iou_indices[i]] in pcls:
                    correct[i] = 1
                    marker[i] = 1
                    pred_uniq_labels.add(labels[best_iou_indices[i]])
                elif pcls[0] in hard_recogs and labels[best_iou_indices[i]] in hard_recogs[pcls[0]]:
                    correct[i] = 1
                    marker[i] = 1
                    pred_uniq_labels.add(labels[best_iou_indices[i]])

                else:
                    wrong_recogs.append((labels[best_iou_indices[i]],cropped,pcls[0]))
                    marker[i] = -1

        
        ignoreable.append(ignore)
        stats.append((correct,
                      best_ious,
                      len(labels),
                      len(uniq_labels),
                      len(pred_uniq_labels),
                      marker))
        predictions.append(prediction_boxes)

    AP = 0
    AR = 0
    aPR = 0
    for (stat_img,ignored) in zip(stats,ignoreable):
        if len(stat_img[0]):
            AR += stat_img[0].sum()/stat_img[2]
            AP += stat_img[0].sum()/( len(stat_img[0]) - ignored)
            aPR += stat_img[4]/stat_img[3]

    print("len of stats",len(stats))
    AP = AP/len(stats)
    AR = AR/len(stats)
    aPR = aPR/len(stats)
    Fscore = (2 * AP * AR) / (AP + AR + epsilon) 
    
    print("[A.Ton] mAP @0.50:      {0:.3f}".format(AP))
    print("[-----] AR @0.50:       {0:.3f}".format(AR))
    print("[-----] AF-score @0.50: {0:.3f}".format(Fscore))
    print("[A.Ton] PR @0.50:       {0:.3f}".format(aPR))
    return predictions,stats,wrong_recogs

@torch.no_grad()
def validate_ISI_format(model,eval_dataset,top_k = 1,plot = False,apply_brisk = False):
    #print("\nStarting evaluation .....\n")
    model.eval()
    
    predictions = []
    ignoreable = []
    
    Pr = []
    Rc = []
    Fsc = []
    for idx in tqdm(range(len(eval_dataset))):
        img_ts,target_ts = eval_dataset[idx]

        img_path = os.path.join(eval_dataset.root, "images", eval_dataset.imgs[idx])
        img_actual = cv2.imread(img_path)
        img_actual = cv2.cvtColor(img_actual,cv2.COLOR_BGR2RGB)
        img_actual = img_actual / 255.0
        img_actual = torch.from_numpy(img_actual).float()
        img_actual = img_actual.permute(2,0,1)

        img_actual = img_actual.to(GPU)
        img_ts = img_ts.to(GPU)
        labels = target_ts["annots"]
        #labels = target_ts["labels"]
        prediction_boxes = model([img_ts])[0]["boxes"].to(CPU)
        n_boxes_pred = prediction_boxes.shape[0]
        TP = 0
        FP = 0
        for pred_box in prediction_boxes:
            tp = 0
            entered = 0
            centre_pred = ((pred_box[0] + pred_box[2])/2 , (pred_box[1] + pred_box[3])/2)

            for (gt_box,label) in zip(target_ts["boxes"],labels):
                X1,Y1,X2,Y2 = gt_box

                if centre_pred[0] >= X1 and centre_pred[0] <= X2 and centre_pred[1] <= Y2 and centre_pred[1] >= Y1:
                    x1,y1,x2,y2 = pred_box
                    
                    x1,y1,x2,y2 = scale_to_coords(x1,y1,x2,y2,img_actual.shape[1:])
                    
                    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                
                    #cropped = img_ts[:,y1:y2,x1:x2]
                    cropped = img_actual[:,y1:y2,x1:x2]
                    pcls = yield_top_k_matches(cropped,k = top_k,apply_BRISK = apply_brisk)

                    entered = 1
                    if label in pcls:
                        tp = 1
                        break

                    if pcls[0] in hard_recogs and label in hard_recogs[pcls[0]]:
                        tp = 1
                        break


            if tp == 1:
                TP += 1
            elif entered == 1:
                FP += 1
        
        Precision = TP/(TP + FP + epsilon)
        Recall = TP /len(labels)
        F_score = (2 * Precision * Recall) /(Precision + Recall + epsilon)

        Pr.append(Precision)
        Rc.append(Recall)
        Fsc.append(F_score)

    AP = sum(Pr)/len(Pr)
    AR = sum(Rc)/len(Rc)
    AFsc = sum(Fsc)/len(Fsc) 
    
    print("\n[ISI] Avg. Precision: {0:.3f}".format(AP))
    print("[ISI] Avg. Recall     {0:.3f}".format(AR))
    print("[ISI] Avg. F-score :  {0:.3f}".format(AFsc))
    return predictions

