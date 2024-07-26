from wavenet_model import *
from audio_data import WavenetDataset
import torch
import torch.nn as nn
import os
import onnxruntime
import numpy as np
import onnx
import shutil
import vai_q_onnx
import soundfile as sf
import argparse

def generate_dataset():
    dtype = torch.FloatTensor
    use_cuda = torch.cuda.is_available()

    model = WaveNetModel(layers=10,
                         blocks=3,
                         dilation_channels=32,
                         residual_channels=32,
                         skip_channels=1024,
                         end_channels=512, 
                         output_length=16,
                         dtype=dtype, 
                         bias=True)
    model = load_latest_model_from('snapshots', use_cuda=use_cuda)

    data = WavenetDataset(dataset_file='train_samples/bach_chaconne/dataset.npz',
                          item_length=model.receptive_field + model.output_length - 1,
                          target_length=model.output_length,
                          file_location='train_samples/bach_chaconne',
                          test_stride=500)
    return model, data

def extract_features(model, data, indices):
    # The input data is based on the sampled audio from data set
    input_length = model.receptive_field + model.output_length - 1
    sample_num = len(indices)

    start_data = torch.zeros((model.classes, input_length))
    selected_indices = indices

    slices = np.linspace(start=0, stop=input_length, num=sample_num + 1).astype(int)

    for i, index in enumerate(selected_indices):
        start_index = slices[i]
        end_index = slices[i + 1]
        start_data[:, start_index:end_index] = data[index][0][:, start_index:end_index]

    start_data = torch.max(start_data, 0)[1] # convert one hot vectors to integers
    return start_data

def model_inference(model_path, config_file_path="vaip_config.json", cache_directory='cache', cacheKey='wavenet_cache'):
    # Specify the path to the quantized ONNZ Model
    onnx_model = onnx.load(model_path)

    # We want to make sure we compile everytime, otherwise the tools will use the cached version
    # Get the current working directory
    current_directory = os.getcwd()
    directory_path = os.path.join(current_directory,  cache_directory, cacheKey)
    cache_directory = os.path.join(current_directory,  cache_directory)

    # Check if the directory exists and delete it if it does.
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
        print(f"Directory deleted successfully. Starting Fresh.")
    else:
        print(f"Directory '{directory_path}' does not exist.")

    aie_options = onnxruntime.SessionOptions()

    aie_session = onnxruntime.InferenceSession(
        onnx_model.SerializeToString(),
        providers=['VitisAIExecutionProvider'],
        sess_options=aie_options,
        provider_options = [{'config_file': config_file_path,
                            'cacheDir': cache_directory,
                            'cacheKey': cacheKey}]
    )

    return aie_session

def generate(model,
             session,
             num_samples,
             first_samples=None,
             temperature=1.):
    model.eval()
    if first_samples is None:
        first_samples = model.dtype(1).zero_()
    generated = Variable(first_samples, volatile=True)

    num_pad = model.receptive_field - generated.size(0)
    if num_pad > 0:
        generated = constant_pad_1d(generated, model.scope, pad_start=True)
        print("pad zero")

    for i in range(num_samples):
        input = Variable(torch.FloatTensor(1, model.classes, model.receptive_field).zero_())
        input = input.scatter_(1, generated[-model.receptive_field:].view(1, -1, model.receptive_field), 1.)

        x = torch.tensor(session.run(None, {"input": input.numpy()})[0])[:, :, -1].squeeze()

        if temperature > 0:
            x /= temperature
            prob = F.softmax(x, dim=0)
            prob = prob.cpu()
            np_prob = prob.data.numpy()
            x = np.random.choice(model.classes, p=np_prob)
            x = Variable(torch.LongTensor([x]))#np.array([x])
        else:
            x = torch.max(x, 0)[1].float()

        generated = torch.cat((generated, x), 0)

    generated = (generated / model.classes) * 2. - 1
    mu_gen = mu_law_expansion(generated, model.classes)

    model.train()
    return mu_gen


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WhisperWave to generate ASMR audios.')
    parser.add_argument('--model_path',
                        type=str,
                        default=r'./models/wavenet.onnx',
                        help='path to ONNX model')
    parser.add_argument('--input', nargs="+", type=int,
                        help='indices of selected audios in the dataset.')
    parser.add_argument('--output',
                        type=str,
                        default=r'./wav/generated_clip.wav',
                        help='path to output wav file')
    parser.add_argument('--num_samples', default=160000, type=int, metavar='N',
                        help='number of generated samples')
    parser.add_argument('--rate', default=16000, type=int, metavar='N',
                        help='rate of generated audio')
    args = parser.parse_args()

    model, data = generate_dataset()
    start_data = extract_features(model, data, args.input)
    aie_session = model_inference(args.model_path)
    generated = generate(model=model,
                         num_samples=args.num_samples,
                         first_samples=start_data,
                         temperature=1.0,
                         session=aie_session)
    sf.write(args.output, generated, args.rate)
    