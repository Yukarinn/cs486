import re

def parse_logs_sketchrnn(filename: str):
    steps = []
    recon_losses = []
    with open(filename) as f:
        lines = [line.strip() for line in f]
        for line in lines:
            vals = re.findall('[\d\.]+', line)
            assert(len(vals) == 7)
            step = int(vals[0])
            lr = float(vals[1])
            klw = float(vals[2])
            cost = float(vals[3])
            recon_loss = float(vals[4])
            kl = float(vals[5])
            train_time_taken = float(vals[6])
            steps.append(step)
            recon_losses.append(recon_loss)
    return steps, recon_losses


def parse_logs_pix2seq(filename: str):
    steps = []
    recon_losses = []
    with open(filename) as f:
        lines = [line.strip() for line in f]
        for line in lines:
            vals = re.findall('[\d\.]+', line)
            assert (len(vals) == 5)
            step = int(vals[0])
            lr = float(vals[1])
            cost = float(vals[2])
            recon_loss = float(vals[3])
            train_time_taken = float(vals[4])
            steps.append(step)
            recon_losses.append(recon_loss)
    return steps, recon_losses
