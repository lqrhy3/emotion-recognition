from torchsummary import summary
from tiny_yolo_model import TinyYolo
import torch
from fpdf import FPDF
import matplotlib.pyplot as plt

PATH_TO_LOG = 'log/logger.log'

coordinate_loss = []
wh_loss = []
conf_loss = []
noobj_loss = []
class_prob = []
total_loss = []

info = ''

with open(PATH_TO_LOG, 'r') as f:
    for number, line in enumerate(f):
        if number in range(2, 10):
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

plt.plot(total_loss)
plt.legend(["TinyYOLO total loss"])
plt.xlabel("Epoch")
plt.ylabel("Total loss")
plt.savefig("total_loss.png")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TinyYolo(grid_size=7, num_bboxes=2)
model = model.to(device)
model_summary = summary(model, (3, 448, 448))

pdf = FPDF()
pdf.add_page()
pdf.set_font("times", size=15, style="B")
pdf.cell(200, 7, txt='Train Report', ln=1, align="C")


pdf.set_font("times", size=12, style="B")
pdf.cell(200, 7, txt='Train info:', ln=1, align="L")
pdf.set_font("courier", size=8)
for line in info.splitlines():
    pdf.cell(200, 3, txt=line, ln=1, align="L")


pdf.set_font("times", size=12, style="B")
pdf.cell(200, 7, txt='Model summary:', ln=1, align="L")
pdf.set_font("courier", size=8)
for line in model_summary.splitlines():
    pdf.cell(200, 3, txt=line, ln=1, align="L")


pdf.set_font("times", size=12, style="B")
pdf.cell(200, 7, txt='Loss:', ln=1, align="L")
pdf.set_font("courier", size=8)
pdf.cell(200, 3, txt="Total loss on the last epoch: " + str(total_loss[-1]), ln=1, align="L")
pdf.image("total_loss.png", x=50, y=190, w=100)

pdf.output("simple_demo.pdf")
