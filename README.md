# Fake News Detection with Text-Image Mismatch Score

**The final project of Stanford CS230 courses (https://cs230.stanford.edu).**

The rapid spread of misinformation and fake news on online platforms poses a significant threat to public trust, social cohesion, and informed decision-making. As multimodal posts combining text and images become more common, they often use striking visuals and emotionally charged headlines to boost engagement, making them more persuasive and difficult to detect with text-only models. Traditional fake news detection models focus mainly on textual analysis, overlooking the visual cues often present in misinformation. Recent studies show that combining text and image signals improves detection accuracy. We therefore aim to develop a multimodal model that jointly leverages both modalities to better assess the veracity of online posts. 

The objective of this project is to design and evaluate a neural network model that can determine whether a given multimodal post (image + text) is likely to represent fake news. Each input instance consists of an image and an associated text caption or title describing the post content. Our project leverages the BLIP-2 Vision–Language Model (VLM) to explicitly model semantic consistency between text and image, enabling deep cross-modal understanding rather than treating the two modalities separately. 


-----

**Orginal Data github repo: https://github.com/entitize/Fakeddit**

Train data images: https://drive.google.com/file/d/1EaSgAheEGBHWQsysokz-Pw6-k9NQUaHr/view?usp=share_link. Mapped via the 'id' column in the csv file.

Dev data images: https://drive.google.com/file/d/1FSvK1CPIt6CqUXR-g4ndp7TbPUazXwpR/view?usp=share_link

Label Structure:
- [2-way] 0 True; 1 False
- [3-way] 0 True; 1 Fake with true text; 2 Fake with false text
- [6-way] 0 True; 1 Satire/Parody; 2 Misleading Content; 3 Imposter Content; 4 False Connection; 5 Manipulated Content 
