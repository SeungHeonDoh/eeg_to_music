from setuptools import setup

setup(
    name="eeg_to_music",
    packages=["eeg_to_music"],
    install_requires=[
        'youtube-dl==2021.6.6',
        'transformers==4.12.5',
        'huggingface-hub==0.2.1',
        'numpy==1.17.3',
        'librosa >= 0.8',
        'pytorch-lightning', # important this version!
        'torchaudio_augmentations==0.2.1', # for augmentation
        'audiomentations==0.22.0',
        'einops',
        'sklearn',
        'wandb',
        'gensim==3.8.3',
        'umap-learn==0.5.2',
        'gradio',
        'pandas',
        'omegaconf',
        "jupyter"
    ]
)