import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    balanced_accuracy_score,
    jaccard_score,
    f1_score,
)

import torch

from utili.misc.misc import compute_output_padding


def eval_metrics(cnn, model_type, Test_loder, DA_Datamodule=None, GPU_NUM=None):
    all_ypre = []
    all_y = []
    for batch in Test_loder:
        x, y = batch

        y_seq = cnn.Fixed_large_model.patchify(y[:, None, None])
        y = cnn.Fixed_large_model.unpatchify(y_seq.clone(), Target=True).squeeze(1).squeeze(1)

        model_out = cnn(x_t=x.cuda(), mask_ratio_t=0, Seg_output=True)

        out = model_out
        out = torch.max(out, 1)[1].squeeze() + 1

        y = torch.squeeze(y)
        out = out[y != -100]
        y = y[y != -100]
        all_ypre.append(out)
        all_y.append(y)
    out = torch.cat(all_ypre, 0)
    y = torch.cat(all_y, 0)

    true_unique_classes = np.unique(y)
    unique_classes = true_unique_classes
    predicted_unique_classes = np.unique(out.cpu())
    true_unique_classes = unique_classes[unique_classes != -100]
    predicted_unique_classes = predicted_unique_classes[predicted_unique_classes != -100]
    union_unique_classes = np.union1d(true_unique_classes, predicted_unique_classes)

    cm = confusion_matrix(
        y.cpu().detach().numpy().astype(np.int64),
        out.cpu().detach().numpy(),
        normalize='true',
    )

    labels_dip = DA_Datamodule.classes_labels_array[:-1]
    fig, ax = plt.subplots(figsize=(15, 15))
    original_classes = DA_Datamodule.original_unique_classes
    display_labels_with_map = labels_dip[
        original_classes[true_unique_classes.astype(np.int64)].astype(np.int64)
    ]
    ConfusionMatrixDisplay.from_predictions(
        y.cpu().detach().numpy().astype(np.int64),
        out.cpu().detach().numpy(),
        normalize='true',
        display_labels=display_labels_with_map,
        xticks_rotation=45,
        ax=ax,
        labels=true_unique_classes.astype(np.int64),
    )
    plt.show()

    y_np = y.cpu().detach().numpy()
    out_np = out.cpu().detach().numpy()
    print(f'{model_type} Acc= {balanced_accuracy_score(y_np, out_np)}')
    print(f'{model_type} mF1= {f1_score(y_np, out_np, average="macro")}')

    return (
        balanced_accuracy_score(y_np, out_np),
        cm,
        jaccard_score(y_np, out_np, average='macro'),
        f1_score(y_np, out_np, average='macro'),
        union_unique_classes,
        f1_score(y_np, out_np, average=None),
    )


def plot_confusion(cm_list, model_name, DA_Datamodule=None, list_f1=None, union_classes=None):
    all_cm = np.mean(np.concatenate(tuple(cm_list), axis=2), 2)
    all_f1 = np.mean(np.stack(tuple(list_f1)), 0)
    labels_dip = DA_Datamodule.classes_labels_array[:]
    original_classes = DA_Datamodule.original_unique_classes

    cm_display = ConfusionMatrixDisplay(
        np.squeeze(all_cm),
        display_labels=labels_dip[original_classes[union_classes.astype(np.int64)].astype(np.int64)],
    )

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_title(model_name)
    cm_display.plot(ax=ax)
    plt.show()
    return all_cm, all_f1


def inference_map(S_cnn=None, T_cnn=None, Target_test_dataloader=None,
                  DA_Datamodule=None, name=None, GPU_NUM=None):
    uh_map = DA_Datamodule.color_map
    values_all = DA_Datamodule.classes_labels_array
    original_classes = DA_Datamodule.original_unique_classes
    gt = torch.tensor(DA_Datamodule.Full_GT_mask)

    for i, batch in enumerate(Target_test_dataloader):
        x, y = batch
        fig, ax = plt.subplots(1, x.shape[0], figsize=(20, 20), layout="tight", squeeze=False)
        out = T_cnn.eval()(x_t=x.clone().cuda(), mask_ratio_t=0, Seg_output=True)
        out = torch.max(out, 1)[1].squeeze() + 1
        gt_y = y
        gt_seq = T_cnn.Fixed_large_model.patchify(gt[None, None, None])
        gt_seq_y = T_cnn.Fixed_large_model.patchify(gt_y[None, None])

        gt = T_cnn.Fixed_large_model.unpatchify(gt_seq.clone(), Target=True).squeeze(1).squeeze(1)
        gt_y = T_cnn.Fixed_large_model.unpatchify(gt_seq_y.clone(), Target=True).squeeze(1).squeeze(1)
        gt[gt == -100] = 0
        gt_y[gt_y == -100] = 0

        mapped_out = original_classes[out.detach().cpu().numpy().astype(np.int64)]
        mapped_out[torch.squeeze(gt) == 0] = 0

        ax[0, i].imshow(
            mapped_out.squeeze().astype(np.int64),
            cmap=uh_map,
            norm=matplotlib.colors.NoNorm(),
        )
        ax[0, i].axis('off')

        selected_original_classes = original_classes[np.unique(gt.detach().cpu().numpy().astype(np.int64))]
        selected_original_classes[selected_original_classes == -100] = values_all.size - 1
        fig.savefig('Inference_map_' + name + '.png', dpi=300, bbox_inches='tight', format='png')

    plt.show()
