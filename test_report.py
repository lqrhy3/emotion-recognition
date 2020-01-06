from torchsummary import summary
import torch
import matplotlib.pyplot as plt
from fpdf import FPDF
import os


PATH_TO_LOG = 'log/emorec/20.01.05_03-09/'

input_shape = (3, 448, 448)
coordinate_loss = []
wh_loss = []
conf_loss = []
noobj_loss = []
class_prob = []
total_loss = []  # Total train loss for Yolo, train loss for others
valid_loss = []
valid_metrics = []

line_for_image = 5


model = torch.load(PATH_TO_LOG + 'model.pt')
model_name = model.__class__.__name__
model_summary = summary(model, input_shape)


# Parsing
with open(os.path.join(PATH_TO_LOG, 'logger.log'), 'r') as f:
    text = f.read()
    epochs = text.split('Epoch:')
    info = epochs[0]

    for epoch in epochs:
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
plt.plot(valid_loss, "r")
plt.plot(total_loss, "g")
plt.legend(["Validation loss", "Train loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(2, 1, 2)
plt.plot(valid_metrics)
plt.legend(["Validation metric"])
plt.xlabel("Epoch")
plt.ylabel("Validation metric")

plt.savefig("graphs.png")


pdf = FPDF()
pdf.add_page()
pdf.set_font("times", size=15, style="B")
pdf.cell(200, 7, txt='{} Train Report'.format(model_name), ln=1, align="C")
line_for_image += 7

pdf.set_font("times", size=12, style="B")
pdf.cell(200, 7, txt='Train info:', ln=1, align="L")
line_for_image += 7
pdf.set_font("courier", size=8)
for line in info.splitlines():
    pdf.cell(200, 3, txt=line, ln=1, align="L")
    line_for_image += 3


pdf.set_font("times", size=12, style="B")
pdf.cell(200, 7, txt='Model summary:', ln=1, align="L")
line_for_image += 7
pdf.set_font("courier", size=8)

for line in model_summary.splitlines():
    pdf.cell(200, 3, txt=line, ln=1, align="C")
    line_for_image += 3

pdf.set_font("times", size=12, style="B")
pdf.cell(200, 7, txt='Results:', ln=1, align="L")
line_for_image += 7
pdf.image('graphs.png', x=55, y=line_for_image, w=120)

pdf.output('report.pdf', dest=PATH_TO_LOG)

