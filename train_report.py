from torchsummary import summary
from tiny_yolo_model import TinyYolo
import torch
from fpdf import FPDF
import matplotlib.pyplot as plt

PATH_TO_LOG = 'log/20.01.03_13-20/logger.log'

coordinate_loss = []
wh_loss = []
conf_loss = []
noobj_loss = []
class_prob = []
total_loss = []
valid_loss = []
valid_iou = []

info = ''
line_for_image = 5

with open(PATH_TO_LOG, 'r') as f:
    for number, line in enumerate(f):
        if number in range(2, 18) and number != 3:
            if str(line) == "___\n":
                info += "\n"
            elif line.find("params") != -1:
                info += line.split(", \'params\'")[0] + "}\n"
            else:
                info += line

        # if line.find('Coordinates loss') != -1:
        #     coordinate_loss.append(float(line.split(':')[1]))
        # if line.find('Width/Height loss') != -1:
        #     wh_loss.append(float(line.split(':')[1]))
        # if line.find('Confidence loss') != -1:
        #     conf_loss.append(float(line.split(':')[1]))
        # if line.find('No object loss') != -1:
        #     noobj_loss.append(float(line.split(':')[1]))
        # if line.find('Class probabilities loss') != -1:
        #     class_prob.append(float(line.split(':')[1]))

        if line.find('Total loss') != -1:
            total_loss.append(float(line.split(':')[1]))
        if line.find('Validation loss') != -1:
            valid_loss.append(float(line.split(':')[1]))
        if line.find('Validation IoU') != -1:
            valid_iou.append(float(line.split(':')[1]))


plt.subplot(2, 1, 1)
plt.plot(valid_loss, "r")
plt.plot(total_loss, "g")
plt.legend(["TinyYOLO validation loss", "TinyYOLO train loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(2, 1, 2)
plt.plot(valid_iou)
plt.legend(["TinyYOLO validation iou"])
plt.xlabel("Epoch")
plt.ylabel("Validation iou")

plt.savefig("graphs.png")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TinyYolo(grid_size=7, num_bboxes=2)
model = model.to(device)
model_summary = summary(model, (3, 448, 448))

pdf = FPDF()
pdf.add_page()
pdf.set_font("times", size=15, style="B")
pdf.cell(200, 7, txt='Train Report', ln=1, align="C")
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
pdf.image("graphs.png", x=55, y=line_for_image, w=120)

pdf.output("simple_demo.pdf")
