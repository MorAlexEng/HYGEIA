
# **Project Title**

Health Yielding General Estimation & Intelligent Analysis (HYGEIA) - Real-Time Heart Rate, Age, Gender & Emotional State Estimation using Rpi and Pi Camera.


## **Description**

This project stems from my MSc thesis in Biomedical Engineering, titled "A Novel Reflective Display Platform for Enhancing Remote Health Monitoring in Home-Based Healthcare." It is named after the ancient Greek goddess of health, Hygeia, and it focuses on the development of a smart mirror system that leverages computer vision to unobtrusively monitor a user's general appearance. By providing real-time audiovisual feedback, the platform supports discreet and continuous health assessment in home-based settings, promoting early detection and preventative care.


## **Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [Credits](#credits)
- [Issues](#issues)
- [Contribute](#contribute)
- [Contact](#contact)
---
## **Installation**

Step-by-step instructions on how to install and set up the project locally. Include any dependences, commands, or environment variables required.

  Clone and download this repository and install its dependencies.
```bash
# Clone the repository
git clone https://github.com/MorAlexEng/HYGEIA

#Navigate to the project directory
cd HYGEIA

#Install dependencies
pip install 'requirements.txt'
```
---

---
## **Usage**

 Connect your Pi Camera (IMX219) to your Raspberry Pi. Once installed test your camera response and brightness.

 ```bash
 raspistill -o Desktop/testimage.jpg
 ```

 Run the python script on your Raspberry Pi.
 ```bash
 python main.py
 ```
---

---
## **Credits**
- [Scientific Literature](https://www.researchgate.net/publication/306285292_Remote_heart_rate_measurement_using_low-cost_RGB_face_video_A_technical_literature_review) for evaluating the use of rPPG for heart rate estimation in remote health monitoring.
- [Methodology Framework](https://link.springer.com/article/10.3758/s13428-024-02398-0) for open-source rPPG applications.
- [Similar project](https://github.com/ganeshkumartk/heartpi) that helped me build my heart rate estimation implementation using rPPG.
- [Another project](https://github.com/bughht/Realtime-rPPG-Application) that helped me build the heart rate estimation algorithm.
- [Remote Heart Rate Estimation](https://github.com/yllberisha/Heart-Rate-EVM) using EVM video processing technique developed by MIT.
- [Age Estimation](https://www.kaggle.com/datasets/alfredhhw/adiencegender) datasest that helped me build my age estimation implementation.
- [Gender Estimation](https://www.kaggle.com/datasets/jangedoo/utkface-new) dataset that helped me build my gender estimation implementation.
- [Emotional State Recognition](https://github.com/berksudan/Real-time-Emotion-Detection) project that helped me build my emotional state recognition implementation.
---

---
## **Issues**

If you face any issues, please pour them [here](https://github.com/MorAlexEng/HYGEIA/issues).

---

---
## **Contribute**

If you would like to contribute to this project with upgrades, fixes or anything, please pull out [Pull Requests](https://github.com/MorAlexEng/HYGEIA/pulls) or contact me.

---

---
## **Contact**

Contact me via e-mail at [alexandrosmor@hotmail.com](mailto:alexandrosmor@hotmail.com)

---
