import torch
from model import PhasefinderModelNoattn
from val import test_model_f_measure
import argparse

def main(modelname):
    datapath = '../stft_db_b_phase_cleaned.h5'

    beat_model = PhasefinderModelNoattn().cuda()
    beat_model.load_state_dict(torch.load(modelname, map_location=torch.device('cuda'), weights_only=True), strict=False)
    beat_model.eval()

    test_model_f_measure(beat_model, datapath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test PhasefinderModel')
    parser.add_argument('modelname', type=str, help='Path to the model file')
    args = parser.parse_args()
    main(args.modelname)
