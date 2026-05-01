# Real-Time ASL Recognition and Bidirectional Learning System

### Abstract

American Sign Language (ASL) serves as the primary mode of communication for millions of
people in the deaf community, yet the barrier between ASL users and people who do not know
it remains a significant challenge in everyday interactions. This project addresses the
communication gap with a real-time, bidirectional ASL recognition system that uses computer
vision and deep learning for seamless translation between sign language and text. The system
utilizes MediaPipe for hand-landmark detection through live input, feeding gesture data into a
custom-trained PyTorch neural network for ASL gesture classification. A Streamlit interface
provides easy usage, making communication accessible in both directions. The result is an
interactive application that demonstrates the power of AI-driven assistive technology in
bridging the communication divide between ASL users and people who do not know it.

Dataset Link: https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data

### How to Run

1. Clone the repository

```bash
git clone https://github.com/raghuram-duvvuri/asl_recognition
cd asl_recognition
```

2. Use Python 3.10 and create a virtual environment

```bash
python3.10 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run the app

```bash
streamlit run web_integration.py
```
