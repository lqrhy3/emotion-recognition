from utils.summary import summary
import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from fpdf import FPDF
import os
import sys


def make_report(PATH_TO_LOG, input_shape):

    total_loss = []  # Total train loss for Yolo, train loss for others
    valid_loss = []
    valid_metrics = []

    line_for_image = 5

    model = torch.load(os.path.join(PATH_TO_LOG, 'model.pt'))
    model_name = model.__class__.__name__
    model_summary = summary(model, input_shape)
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
    plt.plot(valid_loss, 'r')
    plt.plot(total_loss, 'g')
    plt.legend(['Validation loss', 'Train loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(valid_metrics)
    plt.legend(['Validation metric'])
    plt.xlabel('Epoch')
    plt.ylabel('Validation metric')

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

    pdf.output(os.path.join(PATH_TO_LOG, 'report.pdf'))


def find_last_dir():
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
    if len(sys.argv) == 2:
        PATH_TO_LOGDIR = sys.argv[1]
        input_size = 448
    elif len(sys.argv) == 3:
        PATH_TO_LOGDIR = sys.argv[1]
        input_size = int(sys.argv[2])
    else:
        PATH_TO_LOGDIR = find_last_dir()
        input_size = 448

    make_report(PATH_TO_LOGDIR, (3, 320, 320))
