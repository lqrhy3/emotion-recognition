import cv2
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
import time
import torch
from fpdf import FPDF
from utils.summary import summary
from utils.transforms import ImageToTensor


def get_inference_time(model, pth_to_data, input_shape):
    """:param model: model weights
       :param pth_to_data: path to data
       :return inference_time: model inference time in seconds"""
    image = cv2.imread(pth_to_data)
    image = cv2.resize(image, input_shape[1:])
    image = ImageToTensor()(image)
    image = image.unsqueeze(0)

    start = time.time_ns()
    model(image)
    inference_time = time.time_ns() - start

    return inference_time / 1e9


def make_report(PATH_TO_LOG, input_shape):
    """Save .pdf report file about models with an model architecture and parameters
    :param PATH_TO_LOG: path to directory with log of model training
    :param input_shape: size of input image in px"""
    total_loss = []  # Total train loss for Yolo or train loss for others
    valid_loss = []
    valid_metrics = []

    line_for_image = 5

    load = torch.load(os.path.join(PATH_TO_LOG, 'checkpoint.pt'))
    model = torch.load(os.path.join(PATH_TO_LOG, 'model.pt'))
    model.load_state_dict(load['model_state_dict'])
    model.to(torch.device('cpu'))

    # inference_time = get_inference_time(model,
    #                                     pth_to_data='data/detection/train_images_v2/0_Parade_marchingband_1_732.jpg',
    #                                     input_shape=input_shape)
    model_name = model.__class__.__name__
    model_summary = summary(model, input_shape, device='cpu')
    model_summary += '----------------------------------------------------------------\n'
    #model_summary += 'Estimated inference time (seconds per image): ' + str(inference_time) + '\n'
    log_file = ''

    for file in os.listdir(PATH_TO_LOG):
        if file.endswith('.log'):
            log_file = file

    if log_file == '':
        raise RuntimeError('".log" file not found')

    # Parsing
    with open(os.path.join(PATH_TO_LOG, log_file), 'r') as f:
        text = f.read()
        epochs = text.split('Epoch:')
        info = epochs[0]

        for epoch in epochs[1::]:
            phases = epoch.split('\n\n')
            # Train Loss
            if phases[0].find('Total loss') != -1:
                total_loss.append(float(phases[0].split('Total loss:')[1]))
            else:
                total_loss.append(float(phases[0].split('loss:')[1]))

            # Validation Loss
            if phases[1].find('Total loss') != -1:
                valid_loss.append(float(phases[1].split('Total loss:')[1]))
            else:
                valid_loss.append(float(phases[1].split('loss:')[1]))

            # Validation metrics
            if phases[2].find('metrics') != -1:
                valid_metrics.append(float(phases[2].split('metrics:')[1]))

    info = info.replace('Train info:\n', '')

    # Writing to report
    plt.subplot(2, 1, 1)
    plt.grid()
    plt.plot(valid_loss)
    plt.plot(total_loss)
    plt.legend(['Validation loss', 'Train loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.grid()
    plt.plot(valid_metrics, c='crimson')
    plt.legend(['Validation metrics'])
    plt.xlabel('Epoch')
    plt.ylabel('Validation metrics')

    plt.savefig(os.path.join(PATH_TO_LOG, 'graphs.png'))

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('times', size=15, style='B')
    pdf.cell(200, 7, txt='{} Train Report'.format(model_name), ln=1, align='C')
    line_for_image += 7

    pdf.set_font('times', size=12, style='B')
    pdf.cell(200, 7, txt='Model summary:', ln=1, align='L')
    line_for_image += 7
    pdf.set_font('courier', size=8)

    for line in model_summary.splitlines():
        pdf.cell(200, 3, txt=line, ln=1, align='C')
        line_for_image += 3

    pdf.set_font('times', size=12, style='B')
    pdf.cell(200, 7, txt='Train info:', ln=1, align='L')
    line_for_image += 7
    pdf.set_font('courier', size=8)
    for line in info.splitlines():
        pdf.cell(200, 3, txt=line, ln=1, align='L')
        line_for_image += 3

    pdf.set_font('times', size=12, style='B')
    pdf.cell(200, 7, txt='Results:', ln=1, align='L')
    line_for_image += 7
    pdf.image(os.path.join(PATH_TO_LOG, 'graphs.png'), x=45, y=line_for_image, w=100)

    pdf.add_page()

    pdf.set_font('courier', size=8)
    pdf.cell(200, 3, txt=f'Train loss on the last epoch: {total_loss[-1]}', ln=1, align='L')
    pdf.cell(200, 3, txt=f'Validation loss on the last epoch: {valid_loss[-1]} ', ln=1, align='L')
    pdf.cell(200, 3, txt=f'Validation metrics on the last epoch: {valid_metrics[-1]} ', ln=1, align='L')
    line_for_image += 20

    if line_for_image > 266:
        line_for_image = line_for_image % 266
    elif line_for_image > 230:
        pdf.add_page()
        line_for_image = 10

    pdf.image(os.path.join(PATH_TO_LOG, 'graphs.png'), x=45, y=line_for_image, w=100)
    pdf.output(os.path.join(PATH_TO_LOG, 'report.pdf'))


def find_last_dir():
    """:return last_dirname: path to last-edited directory in folder 'log' """
    last_mod_time = 0
    last_dirname = ''
    folders = os.listdir('log/')
    for folder in folders:
        subfolders = os.listdir(os.path.join('log/', folder))
        for subfolder in subfolders:
            folder_name = os.path.join('log/', folder, subfolder)
            mod_time = os.path.getmtime(folder_name)
            if mod_time > last_mod_time:
                last_dirname = folder_name
                last_mod_time = mod_time

    return last_dirname


if __name__ == '__main__':
    """:param sys.argv[1]: path to directory with logs
       :param sys.argv[2] (optional): size of input image"""
    if len(sys.argv) == 2:
        PATH_TO_LOGDIR = sys.argv[1]
        input_shape = 448
    elif len(sys.argv) == 3:
        PATH_TO_LOGDIR = sys.argv[1]
        input_shape = int(sys.argv[2])
    else:
        PATH_TO_LOGDIR = find_last_dir()
        input_shape = 448

    make_report(PATH_TO_LOGDIR, (1, 64, 64))
