import torch
import os
import argparse
import sys
import copy

from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification


from pytorchvideo.transforms import ApplyTransformToKey, Normalize
from pytorchvideo.data.encoded_video import EncodedVideo

from torchvision.transforms import Compose, Lambda, Resize


model_ckpt = "MCG-NJU/videomae-base" # pre-trained model from which to fine-tune

class DecoderMAE(torch.nn.Module):
    def __init__(self, model_ckpt):
        super(DecoderMAE, self).__init__()
        self.videoMAE = VideoMAEForVideoClassification.from_pretrained(
                        model_ckpt,
                        ignore_mismatched_sizes=True,
                        num_labels=768,
                        output_hidden_states=True,
                        return_dict=True,
                    )
        self.decoder = torch.nn.Linear(768, 1568*768)
        
    def forward(self, x):
        x = self.videoMAE(x, output_hidden_states=True)
        hidden_states = x.hidden_states[-1]
        # print(x.keys(), x.hidden_states[-1].shape)
        x = self.decoder(x.logits)
        x = x.reshape(x.shape[0], 1568, 768)
        return x, hidden_states


class DatasetVideoMae(torch.utils.data.Dataset):
    def __init__(self, root_path): 
        self.root_path = root_path
        model_ckpt = "MCG-NJU/videomae-base" # pre-trained model from which to fine-tune
        image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
        
        mean = image_processor.image_mean
        std = image_processor.image_std
        if "shortest_edge" in image_processor.size:
            height = width = image_processor.size["shortest_edge"]
        else:
            height = image_processor.size["height"]
            width = image_processor.size["width"]
        resize_to = (height, width)
        
        self.transform = Compose(
                        [
                            ApplyTransformToKey(
                                key="video",
                                transform=Compose(
                                    [
                                        # UniformTemporalSubsample(num_frames_to_sample),
                                        Lambda(lambda x: x / 255.0),
                                        Normalize(mean, std),
                                        Resize(resize_to, antialias=True),
                                    ]
                                ),
                            ),
                        ]
                    )

        self.videos_names = os.listdir(self.root_path)
        self.load_videos()
        self.counter = {k: 0 for k in self.videos_names}

    def __len__(self):
        return sum([v["video"].shape[1]-16 for v in self.videos])

    def load_videos(self):
        self.videos = []
        for name in self.videos_names:
            video = EncodedVideo.from_path(os.path.join(self.root_path, name))
            video = video.get_clip(0.0, 1000000)
            
            video["name"] = name
            self.videos.append(video)
        
    def __getitem__(self, idx):
        i = 0
        c = 0
        while c < idx:
            if c+self.videos[i]["video"].shape[1]-16 < idx: 
                c += self.videos[i]["video"].shape[1]-16
                i += 1
            else: 
                break
        
        video = copy.deepcopy(self.videos[i])
        start = self.counter[self.videos_names[i]]
        self.counter[self.videos_names[i]] += 1

        video["video"] = video["video"][:, start:start+16, :, :]
        video = self.transform(video)
        video["video"] = video["video"].permute(1, 0, 2, 3)
        return {"name": video["name"], "video": video["video"], "start": start} 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess image sequence')
    parser.add_argument('--path2DAVIS', type=str, help='Path to DAVIS dataset')
    args = parser.parse_args()
    path2VideoMAE_ckpt = os.path.join("checkpoints", "VideoMAE.pth")
    dataset = DatasetVideoMae(root_path=os.path.join(args.path2DAVIS, "MP4"))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DecoderMAE(model_ckpt)
    model.load_state_dict(torch.load(path2VideoMAE_ckpt))
    model.eval()
    model = model.to(device)

    os.makedirs(os.path.join(args.path2DAVIS, "EMBEDDINGS"), exist_ok=True)
    with torch.no_grad():
        for i, inputs in enumerate(dataloader):
            name, video, start = inputs["name"], inputs["video"], inputs["start"]
            video = video.to(device)

            outputs = model.videoMAE(video).logits

            for i, n in enumerate(name):
                torch.save(outputs[i], os.path.join(args.path2DAVIS, "EMBEDDINGS", f"{name[i]}_{start[i]}.pt"))